"""Tick-level backtest engine: simulate trades on real bid/ask tick data.

Execution model (the same conventions a broker's matching engine uses):

- A signal carries a decision time `ts` (UTC epoch ms) — typically the close
  time of the bar that generated it. The strategy cannot act before that.
- At decision time the engine takes the first available quote on each side:
  ``t_dec`` = min(first BID tick >= ts, first ASK tick >= ts). The prices used
  are the last tick <= t_dec on each side (``bid_d``, ``ask_d``).
- BUY fills at ``ask_d``, SELL fills at ``bid_d`` — a market order paying the
  real spread at that instant.
- If ``(ask_d - bid_d) / pip_size`` exceeds ``max_spread_pips`` the signal is
  skipped (configurable; ``inf`` disables the filter).
- While the position is open, exits are evaluated tick-by-tick on the
  CLOSE-side feed: a BUY watches BID ticks (you sell at bid), a SELL watches
  ASK ticks (you buy back at ask).
- Per tick the checks run in order: stop-loss, take-profit, then the
  break-even trigger. If one tick could hit both SL and TP the SL wins —
  intra-tick ambiguity is resolved conservatively.
- Break-even: once price reaches ``entry +/- be_trigger_pips`` the stop moves
  to ``entry +/- be_offset_pips``. If ``partial_lots > 0`` a partial position
  is also closed at market on that tick.
- If the data ends with a position still open it is closed at the last tick
  (reason ``"eod"``).
- One position at a time: signals whose decision time falls inside an open
  trade are skipped.
- Costs are applied per trade: ``commission_per_lot_rt * volume`` plus
  ``pip_value_usd_per_lot * pips * lots`` per exit leg, plus swap for each
  broker-day boundary (server midnight) crossed while holding — see
  ``count_rollovers`` for the timezone model.

The engine itself is strategy-agnostic: it consumes a list of signal dicts
``{"ts": int, "dir": "BUY"|"SELL", "vol": float, "meta": {...}}`` and an
``ExitRules``. How signals are produced (bars, indicators, ML, discretion)
is entirely up to you — see ``examples/``.
"""

import datetime as dt
from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# costs and timezone model
# ---------------------------------------------------------------------------
def fixed_offset(hours):
    """Rollover offset factory: server clock is always UTC+`hours`."""
    return lambda ts_ms: hours


def _last_sunday(year, month):
    d = dt.date(year, month, 28)
    while (d + dt.timedelta(days=1)).month == month:
        d += dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def eu_broker_offset(winter_hours=2, summer_hours=3):
    """UTC offset for brokers on EU time (EET/EEST — Pepperstone, most EU
    brokers): +winter_hours normally, +summer_hours between the last Sunday
    of March and the last Sunday of October at 01:00 UTC.

    Swap is charged once per broker-day boundary (server midnight). If your
    broker uses a different convention, pass ``fixed_offset(h)`` or your own
    ``ts_ms -> hours`` callable to ``CostModel.rollover_offset``.
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


def count_rollovers(t_start_ms, t_end_ms, offset_fn):
    """Count broker-day boundaries crossed in (t_start, t_end].

    A boundary is server midnight = (24 - offset) UTC hour of each day. The
    offset is re-evaluated at each candidate so DST transitions are handled.
    """
    if t_end_ms <= t_start_ms:
        return 0
    n = 0
    t = t_start_ms
    while t < t_end_ms and n < 400:
        off = offset_fn(t)
        prox_h = (24 - off) % 24
        cand = (t // 86_400_000) * 86_400_000 + prox_h * 3_600_000
        if cand <= t:
            cand += 86_400_000
        off2 = offset_fn(cand)
        if off2 != off:
            prox_h = (24 - off2) % 24
            cand = (t // 86_400_000) * 86_400_000 + prox_h * 3_600_000
            if cand <= t:
                cand += 86_400_000
        if cand > t_end_ms:
            break
        n += 1
        t = cand + 1
    return n


@dataclass
class CostModel:
    pip_size: float = 1e-4
    # USD earned per pip per 1.0 lot. Quote-currency dependent:
    # EURUSD ~= 10, XAUUSD ~= 10/oz convention differs, index CFDs vary.
    pip_value_usd_per_lot: float = 10.0
    commission_per_lot_rt: float = 0.0   # round-trip, per 1.0 lot
    swap_per_lot_night_buy: float = 0.0  # per 1.0 lot per broker-day
    swap_per_lot_night_sell: float = 0.0
    rollover_offset: object = None       # ts_ms -> server UTC offset hours
    max_spread_pips: float = float("inf")

    def __post_init__(self):
        if self.rollover_offset is None:
            self.rollover_offset = eu_broker_offset()


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
    decision_ts: int
    entry_ts: int
    entry: float
    exit_ts: int
    volume: float
    pnl_usd: float
    pnl_pips: float          # last exit leg, pips
    reason: str              # "sl" | "tp" | "partial"->see exits | "eod"
    nights: int
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
            tr = self._simulate_one(s, buy, vol, t_dec, bid_d, ask_d,
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
            "reason": t.reason, "nights": t.nights,
        } for t in trades]
        return pd.DataFrame(rows)

    # -- internals ------------------------------------------------------
    def _decision_quote(self, ent_t):
        """First cross-side tick >= ent_t and the prices at that instant."""
        kb = bisect_left(self.bid_ts, ent_t)
        ka = bisect_left(self.ask_ts, ent_t)
        cand = []
        if kb < len(self.bid_ts):
            cand.append(int(self.bid_ts[kb]))
        if ka < len(self.ask_ts):
            cand.append(int(self.ask_ts[ka]))
        if not cand:
            return None, None, None
        t_dec = min(cand)
        ib = bisect_left(self.bid_ts, t_dec) - 1
        ia = bisect_left(self.ask_ts, t_dec) - 1
        if ib < 0 or ia < 0:
            return t_dec, None, None
        return t_dec, float(self.bid_px[ib]), float(self.ask_px[ia])

    def _simulate_one(self, s, buy, vol, t_dec, bid_d, ask_d,
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

        k0 = bisect_left(side_ts, t_dec)
        be_done = partial_done = False
        vol_rest = vol
        legs = []
        j = k0
        n = len(side_ts)
        while j < n:
            t = int(side_ts[j])
            px = float(side_px[j])
            if buy:
                if px <= sl:
                    legs.append((t, min(sl, px), vol_rest, "sl")); break
                if px >= tp:
                    legs.append((t, tp, vol_rest, "tp")); break
                if not be_done and trig_d > 0 and px >= entry + trig_d:
                    be_done = True
                    sl = be_px
                    if exits.partial_lots > 0 and not partial_done and \
                       vol_rest > exits.partial_lots + exits.min_lot - 1e-9:
                        fecha = min(exits.partial_lots, vol_rest - exits.min_lot)
                        if fecha > 0:
                            legs.append((t, px, fecha, "partial"))
                            vol_rest -= fecha
                            partial_done = True
            else:
                if px >= sl:
                    legs.append((t, max(sl, px), vol_rest, "sl")); break
                if px <= tp:
                    legs.append((t, tp, vol_rest, "tp")); break
                if not be_done and trig_d > 0 and px <= entry - trig_d:
                    be_done = True
                    sl = be_px
                    if exits.partial_lots > 0 and not partial_done and \
                       vol_rest > exits.partial_lots + exits.min_lot - 1e-9:
                        fecha = min(exits.partial_lots, vol_rest - exits.min_lot)
                        if fecha > 0:
                            legs.append((t, px, fecha, "partial"))
                            vol_rest -= fecha
                            partial_done = True
            j += 1

        if not legs:
            if j > k0:
                legs.append((int(side_ts[j - 1]), float(side_px[j - 1]),
                             vol_rest, "eod"))
            else:
                return None

        exit_ts = legs[-1][0]
        usd = 0.0
        for (_te, pxe, ve, _mot) in legs:
            pips = (pxe - entry) / pip if buy else (entry - pxe) / pip
            usd += pips * c.pip_value_usd_per_lot * ve
        usd -= c.commission_per_lot_rt * vol
        noches = count_rollovers(t_dec, exit_ts, c.rollover_offset)
        usd += (c.swap_per_lot_night_buy if buy else c.swap_per_lot_night_sell) \
            * vol * noches

        last_px = legs[-1][1]
        pnl_pips = (last_px - entry) / pip if buy else (entry - last_px) / pip
        return Trade(
            direction="BUY" if buy else "SELL",
            decision_ts=t_dec, entry_ts=t_dec, entry=entry,
            exit_ts=exit_ts, volume=vol, pnl_usd=usd, pnl_pips=pnl_pips,
            reason=legs[-1][3], nights=noches, exits=legs,
            meta=dict(s.get("meta") or {}),
        )
