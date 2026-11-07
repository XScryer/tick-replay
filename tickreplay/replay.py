"""Tick-level backtest engine: simulate trades on real bid/ask tick data.

Execution model (deliberately simple and explicit — read it before trusting
any result):

- A signal carries a decision time ``ts`` (UTC epoch ms) — typically the close
  time of the bar that generated it. The strategy cannot act before that.
- The order reaches the market on the first tick of either side at/after
  ``ts``: ``t_dec = min(first BID tick >= ts, first ASK tick >= ts)``. The
  quote used is the latest tick at/before ``t_dec`` on each side
  (``bid_d``, ``ask_d``), so the tick that arrived at ``t_dec`` is included.
- BUY fills at ``ask_d``, SELL fills at ``bid_d`` — a market order paying the
  real spread at that instant. Latency, slippage beyond the quote and
  liquidity/partial fills are not modelled.
- If ``(ask_d - bid_d) / pip_size`` exceeds ``max_spread_pips`` the signal is
  skipped (configurable; ``inf`` disables the filter).
- While the position is open, exits are evaluated tick-by-tick on the
  CLOSE-side feed: a BUY watches BID ticks (you sell at bid), a SELL watches
  ASK ticks (you buy back at ask).
- Per tick the checks run in order: stop-loss, take-profit, then the
  break-even trigger. If one tick could hit both SL and TP the SL wins —
  intra-tick ambiguity is resolved conservatively. A stop that is gapped
  through fills at the (worse) tick price; a take-profit fills at the TP
  level, never better.
- Break-even: once price reaches ``entry +/- be_trigger_pips`` the stop moves
  to ``entry +/- be_offset_pips``. If ``partial_lots > 0`` a partial position
  is also closed at market on that tick.
- If the data ends with a position still open it is closed at the last tick
  (reason ``"eod"``).
- One position at a time: signals whose decision time falls inside an open
  trade are skipped.
- Costs are applied per trade: ``pip_value_usd_per_lot * pips * lots`` per
  exit leg, minus ``commission_per_lot_rt * volume``, plus swap for every
  rollover crossed while holding (server midnight; triple on Wednesday, none
  for the weekend — see ``swap_days`` and ``CostModel``).

The engine itself is strategy-agnostic: it consumes a list of signal dicts
``{"ts": int, "dir": "BUY"|"SELL", "vol": float | callable, "meta": {...}}``
and an ``ExitRules``. How signals are produced (bars, indicators, ML,
discretion) is entirely up to you — see ``examples/``.
"""

import datetime as dt
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

import numpy as np

DAY_MS = 86_400_000
HOUR_MS = 3_600_000


# ---------------------------------------------------------------------------
# broker server clock and rollover (swap) model
# ---------------------------------------------------------------------------
def fixed_offset(hours):
    """Rollover offset factory: server clock is always UTC+``hours``."""
    return lambda ts_ms: hours


def _last_sunday(year, month):
    d = dt.date(year, month, 28)
    while (d + dt.timedelta(days=1)).month == month:
        d += dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def _nth_sunday(year, month, n):
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(6 - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def ny_close_offset(winter_hours=2, summer_hours=3):
    """Server clock of "New York close" brokers (the usual FX convention,
    used e.g. by Pepperstone and IC Markets): server midnight = 17:00 in
    New York, i.e. UTC+2 while the US is on standard time and UTC+3 during
    US daylight saving (second Sunday of March 07:00 UTC to first Sunday of
    November 06:00 UTC). This is the default ``CostModel.rollover_offset``.
    """
    utc = dt.timezone.utc

    def offset(ts_ms):
        t = dt.datetime.fromtimestamp(ts_ms / 1000.0, utc)
        s, e = _nth_sunday(t.year, 3, 2), _nth_sunday(t.year, 11, 1)
        start = dt.datetime(s.year, s.month, s.day, 7, tzinfo=utc)
        end = dt.datetime(e.year, e.month, e.day, 6, tzinfo=utc)
        return summer_hours if start <= t < end else winter_hours

    return offset


def eu_broker_offset(winter_hours=2, summer_hours=3):
    """Server clock that follows EU daylight saving (EET/EEST): UTC+2
    normally, UTC+3 from the last Sunday of March to the last Sunday of
    October at 01:00 UTC. For the few weeks a year when EU and US DST
    differ this puts server midnight one hour away from the NY close — use
    it only if your broker's server time really follows EU rules.
    """
    utc = dt.timezone.utc

    def offset(ts_ms):
        t = dt.datetime.fromtimestamp(ts_ms / 1000.0, utc)
        start = dt.datetime(t.year, 3, _last_sunday(t.year, 3).day, 1,
                            tzinfo=utc)
        end = dt.datetime(t.year, 10, _last_sunday(t.year, 10).day, 1,
                          tzinfo=utc)
        return summer_hours if start <= t < end else winter_hours

    return offset


def rollover_times(t_start_ms, t_end_ms, offset_fn):
    """Server-midnight instants (UTC ms) in ``(t_start_ms, t_end_ms]``.

    ``offset_fn(ts_ms)`` returns the server's UTC offset in hours at that
    instant; it is re-evaluated at every boundary, so DST changes are
    handled.
    """
    if t_end_ms <= t_start_ms:
        return []
    out = []
    d = (t_start_ms // DAY_MS - 1) * DAY_MS
    while d <= t_end_ms + DAY_MS:
        nxt = d + DAY_MS                                   # UTC midnight
        off = offset_fn(nxt)
        b = nxt - int(round(off * HOUR_MS))
        off_b = offset_fn(b)
        if off_b != off:
            b = nxt - int(round(off_b * HOUR_MS))
        if t_start_ms < b <= t_end_ms:
            out.append(b)
        d = nxt
    return out


def count_rollovers(t_start_ms, t_end_ms, offset_fn):
    """Number of server-midnight boundaries crossed in ``(t_start, t_end]``."""
    return len(rollover_times(t_start_ms, t_end_ms, offset_fn))


def _server_weekday(b_ms, offset_fn):
    """Weekday (Mon=0) of the server day that ENDS at boundary ``b_ms``."""
    local = b_ms - 1 + int(round(offset_fn(b_ms - 1) * HOUR_MS))
    return int((local // DAY_MS + 3) % 7)                  # 1970-01-01 = Thu


def swap_days(t_start_ms, t_end_ms, offset_fn, triple_weekday=2):
    """Swap units charged for holding over ``(t_start, t_end]``.

    Spot FX convention: each rollover at server midnight charges one day of
    swap, except that the rollover ending ``triple_weekday`` (default
    Wednesday, Mon=0) charges three to cover the weekend, and the rollovers
    ending Saturday and Sunday charge nothing. So holding Friday -> Monday
    costs 1 day and Wednesday -> Thursday costs 3. cTrader reports the
    triple day per symbol (``ProtoOASymbol.swapRollover3Days``).
    ``triple_weekday=None`` charges every boundary once (7-day markets).
    """
    total = 0
    for b in rollover_times(t_start_ms, t_end_ms, offset_fn):
        if triple_weekday is None:
            total += 1
            continue
        wd = _server_weekday(b, offset_fn)
        if wd == triple_weekday:
            total += 3
        elif wd < 5:
            total += 1
    return total


@dataclass
class CostModel:
    pip_size: float = 1e-4
    # USD earned per pip per 1.0 lot. Quote-currency dependent:
    # EURUSD ~= 10, XAUUSD ~= 10/oz convention differs, index CFDs vary.
    pip_value_usd_per_lot: float = 10.0
    commission_per_lot_rt: float = 0.0   # round-trip, per 1.0 lot
    swap_per_lot_night_buy: float = 0.0  # USD per 1.0 lot per swap day
    swap_per_lot_night_sell: float = 0.0
    rollover_offset: object = None       # ts_ms -> server UTC offset hours
    triple_swap_weekday: object = 2      # Mon=0; None = every night once
    max_spread_pips: float = float("inf")

    def __post_init__(self):
        if self.rollover_offset is None:
            self.rollover_offset = ny_close_offset()


@dataclass
class ExitRules:
    """How an open position leaves the market."""
    stop_pips: float
    take_pips: float
    be_trigger_pips: float = 0.0   # 0 disables break-even
    be_offset_pips: float = 0.0    # stop -> entry +/- this once triggered
    partial_lots: float = 0.0      # >0: close this many lots at market on BE
    min_lot: float = 0.01


@dataclass
class Trade:
    direction: str
    decision_ts: int         # signal ts (when the strategy decided)
    entry_ts: int            # first tick at/after decision_ts (the fill)
    entry: float
    exit_ts: int
    volume: float
    pnl_usd: float
    pnl_pips: float          # last exit leg, pips
    reason: str              # final leg: "sl" | "tp" | "eod" (partials in exits)
    nights: int              # server-midnight rollovers crossed
    swap_days: int = 0       # swap units charged (triple Wednesday, no weekend)
    exits: list = field(default_factory=list)  # (ts, px, lots, reason)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class ReplayEngine:
    """Simulates signal dicts on real tick arrays.

    bid_ts/bid_px, ask_ts/ask_px: int64 ms UTC + float64 prices, sorted.
    """

    def __init__(self, bid_ts, bid_px, ask_ts, ask_px, costs=None):
        self.bid_ts = np.ascontiguousarray(bid_ts, dtype=np.int64)
        self.bid_px = np.ascontiguousarray(bid_px, dtype=np.float64)
        self.ask_ts = np.ascontiguousarray(ask_ts, dtype=np.int64)
        self.ask_px = np.ascontiguousarray(ask_px, dtype=np.float64)
        if len(self.bid_ts) != len(self.bid_px) or \
           len(self.ask_ts) != len(self.ask_px):
            raise ValueError("ts/px length mismatch")
        self.costs = costs or CostModel()

    # -- public ---------------------------------------------------------
    def run(self, signals, exits, deposit=0.0):
        """Execute signals under `exits` rules. Returns list[Trade].

        `deposit` is the starting equity; it is only used when a signal's
        "vol" is a callable — the callable receives
        {"equity": current_balance, "deposit": deposit, "trades": n_done}
        and must return lots, so compounding sizing schemes work.
        """
        pip = self.costs.pip_size
        stop_d = exits.stop_pips * pip
        take_d = exits.take_pips * pip
        trig_d = exits.be_trigger_pips * pip
        be_off = exits.be_offset_pips * pip

        trades = []
        pos_end = 0
        equity = float(deposit)
        for s in sorted(signals, key=lambda x: int(x["ts"])):
            ent_t = int(s["ts"])
            if ent_t <= pos_end:
                continue
            t_dec, bid_d, ask_d = self._decision_quote(ent_t)
            if t_dec is None:
                break                      # no more ticks at all: stop
            if bid_d is None:
                continue                   # not enough history on one side
            if (ask_d - bid_d) / pip > self.costs.max_spread_pips:
                continue
            buy = str(s["dir"]).upper() == "BUY"
            v = s.get("vol", 0.01)
            if callable(v):
                v = v({"equity": equity, "deposit": deposit,
                       "trades": len(trades)})
            vol = round(float(v), 2)
            if vol <= 0:
                continue
            tr = self._simulate_one(s, ent_t, buy, vol, t_dec, bid_d, ask_d,
                                    stop_d, take_d, trig_d, be_off, exits, pip)
            if tr is None:
                continue
            trades.append(tr)
            equity += tr.pnl_usd
            pos_end = tr.exit_ts
        return trades

    def trades_frame(self, trades):
        """Trades as a pandas DataFrame (one row per trade)."""
        import pandas as pd
        rows = [{
            "direction": t.direction, "decision_ts": t.decision_ts,
            "entry_ts": t.entry_ts, "entry": t.entry, "exit_ts": t.exit_ts,
            "volume": t.volume, "pnl_usd": t.pnl_usd, "pnl_pips": t.pnl_pips,
            "reason": t.reason, "nights": t.nights, "swap_days": t.swap_days,
        } for t in trades]
        return pd.DataFrame(rows)

    # -- internals ------------------------------------------------------
    def _decision_quote(self, ent_t):
        """First tick (either side) >= ent_t and the quote at that instant."""
        kb = bisect_left(self.bid_ts, ent_t)
        ka = bisect_left(self.ask_ts, ent_t)
        candidate = []
        if kb < len(self.bid_ts):
            candidate.append(int(self.bid_ts[kb]))
        if ka < len(self.ask_ts):
            candidate.append(int(self.ask_ts[ka]))
        if not candidate:
            return None, None, None
        t_dec = min(candidate)
        ib = bisect_right(self.bid_ts, t_dec) - 1
        ia = bisect_right(self.ask_ts, t_dec) - 1
        if ib < 0 or ia < 0:
            return t_dec, None, None
        return t_dec, float(self.bid_px[ib]), float(self.ask_px[ia])

    def _simulate_one(self, s, ent_t, buy, vol, t_dec, bid_d, ask_d,
                      stop_d, take_d, trig_d, be_off, exits, pip):
        c = self.costs
        if buy:
            entry = ask_d
            sl, tp = entry - stop_d, entry + take_d
            side_ts, side_px = self.bid_ts, self.bid_px
            be_px = entry + be_off
        else:
            entry = bid_d
            sl, tp = entry + stop_d, entry - take_d
            side_ts, side_px = self.ask_ts, self.ask_px
            be_px = entry - be_off

        start_idx = bisect_left(side_ts, t_dec)
        be_done = partial_done = False
        vol_left = vol
        legs = []
        j = start_idx
        n = len(side_ts)
        while j < n:
            t = int(side_ts[j])
            px = float(side_px[j])
            if buy:
                if px <= sl:
                    legs.append((t, min(sl, px), vol_left, "sl")); break
                if px >= tp:
                    legs.append((t, tp, vol_left, "tp")); break
                if not be_done and trig_d > 0 and px >= entry + trig_d:
                    be_done = True
                    sl = be_px
                    if exits.partial_lots > 0 and not partial_done and \
                       vol_left > exits.partial_lots + exits.min_lot - 1e-9:
                        qty = min(exits.partial_lots, vol_left - exits.min_lot)
                        if qty > 0:
                            legs.append((t, px, qty, "partial"))
                            vol_left -= qty
                            partial_done = True
            else:
                if px >= sl:
                    legs.append((t, max(sl, px), vol_left, "sl")); break
                if px <= tp:
                    legs.append((t, tp, vol_left, "tp")); break
                if not be_done and trig_d > 0 and px <= entry - trig_d:
                    be_done = True
                    sl = be_px
                    if exits.partial_lots > 0 and not partial_done and \
                       vol_left > exits.partial_lots + exits.min_lot - 1e-9:
                        qty = min(exits.partial_lots, vol_left - exits.min_lot)
                        if qty > 0:
                            legs.append((t, px, qty, "partial"))
                            vol_left -= qty
                            partial_done = True
            j += 1

        if not legs:
            if j > start_idx:
                legs.append((int(side_ts[j - 1]), float(side_px[j - 1]),
                             vol_left, "eod"))
            else:
                return None

        exit_ts = legs[-1][0]
        usd = 0.0
        for (_texit, pxexit, qexit, _reason) in legs:
            pips = (pxexit - entry) / pip if buy else (entry - pxexit) / pip
            usd += pips * c.pip_value_usd_per_lot * qexit
        usd -= c.commission_per_lot_rt * vol
        rollovers = count_rollovers(t_dec, exit_ts, c.rollover_offset)
        charged = swap_days(t_dec, exit_ts, c.rollover_offset,
                            c.triple_swap_weekday)
        usd += (c.swap_per_lot_night_buy if buy else c.swap_per_lot_night_sell) \
            * vol * charged

        last_px = legs[-1][1]
        pnl_pips = (last_px - entry) / pip if buy else (entry - last_px) / pip
        return Trade(
            direction="BUY" if buy else "SELL",
            decision_ts=ent_t, entry_ts=t_dec, entry=entry,
            exit_ts=exit_ts, volume=vol, pnl_usd=usd, pnl_pips=pnl_pips,
            reason=legs[-1][3], nights=rollovers, swap_days=charged, exits=legs,
            meta=dict(s.get("meta") or {}),
        )
