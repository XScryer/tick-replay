"""Worked example: EURUSD M15 wick-fade strategy on real cTrader ticks.

A complete, runnable reference strategy showing how to drive the engine
from bar-based signals. It is an example of the workflow, NOT a
recommendation: the parameters below are the author's own settings and no
out-of-sample evidence is included in this repository.

  - CyclicRSI (cRSI) on bar closes, two speeds (fast + 4x slow for HTF filter)
  - Entry: rejection-wick fade — long lower wick / short upper wick
  - Session filter: only 14:00-18:00 UTC (NY morning), optional Thursday rule
  - Optional regime gate: skip fades while the market is in a confirmed leg
    (tickreplay.regime.regime_opinion)
  - Exits on ticks: SL / TP / break-even trigger, real spread filter,
    commission and swap (set yours with --commission / --swap-buy / --swap-sell)
  - Position sizing: fixed lot or the compound "ladder" (volume grows with
    equity — implemented through the engine's callable `vol`)

Requires EURUSD tick data in .npz format (see the README quickstart):

    python -m tickreplay.ctrader --symbol EURUSD --from 2024-01-01 --to 2024-12-31 --out data/EURUSD/raw
    python -m tickreplay.data csv2bin --raw-dir data/EURUSD/raw --symbol EURUSD --bin-dir data/EURUSD/bin
    python examples/eurusd_wick_fade.py --bin-dir data/EURUSD/bin --from 2024-02-01 --to 2024-12-31
"""

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from tickreplay.data import build_bars, load_ticks, timeframe_ms
from tickreplay.regime import regime_opinion
from tickreplay.replay import CostModel, ExitRules, ReplayEngine

PIP = 0.0001
DAY_MS = 86_400_000
WARMUP_DAYS = 30        # bar history before --from for indicator warm-up
TAIL_DAYS = 30          # tick history after --to so open trades can close


# ---------------------------------------------------------------------------
# CyclicRSI (cRSI, after Lars von Thienen) — smoothed, phase-shifted RSI
# ---------------------------------------------------------------------------
class CyclicRSI:
    def __init__(self, dom_cycle):
        self.cycle_len = max(dom_cycle // 2, 1)
        self.phasing_lag = 4
        self.torque = 2.0 / 11.0
        self.alpha = 1.0 / self.cycle_len
        self.rma_up = 0.0
        self.rma_down = 0.0
        self.prev_rsi = 50.0
        self.prev_c = 50.0
        self.buf = [50.0] * 200
        self.pos = 0

    def update(self, change):
        up = change if change > 0.0 else 0.0
        down = -change if change < 0.0 else 0.0
        if self.rma_up == 0.0 and self.rma_down == 0.0:
            self.rma_up = up
            self.rma_down = down
        else:
            self.rma_up = self.alpha * up + (1.0 - self.alpha) * self.rma_up
            self.rma_down = self.alpha * down + (1.0 - self.alpha) * self.rma_down
        if self.rma_down == 0.0 and self.rma_up == 0.0:
            rsi = 50.0
        elif self.rma_down == 0.0:
            rsi = 100.0
        elif self.rma_up == 0.0:
            rsi = 0.0
        else:
            rsi = 100.0 - (100.0 / (1.0 + (self.rma_up / self.rma_down)))
        rsi_lag = self.buf[(self.pos - self.phasing_lag) % 200]
        self.buf[self.pos % 200] = rsi
        self.pos += 1
        c = self.torque * (2.0 * rsi - rsi_lag) + (1.0 - self.torque) * self.prev_c
        self.prev_c = c
        self.prev_rsi = rsi
        return c


# ---------------------------------------------------------------------------
# configuration presets (EURUSD M15)
# ---------------------------------------------------------------------------
PRESETS = {
    # with the regime gate (K=12 bars, net move >= 10 pips, efficiency >= 0.45,
    # 6-bar backward consensus) and the Thursday-after-16:00 rule
    "wick_fade_regime": dict(
        stop_pips=10.0, be_trigger_pips=12.0, take_pips=32.0,
        be_offset_pips=0.0, exit_mode=0, partial_size=0.01,
        max_spread_pips=1.5, cycle=24, oversold=22.5, overbought=80.0,
        wick_frac=0.40, htf_filter=True, hour_from=14, hour_to=18,
        days_mask=127, thu_from=16, compound=True, profit_ladder=False,
        ladder_step=50.0, lot_step=0.01, max_lot=0.20, fixed_lot=0.01,
        regime_gate=True, regime_K=12, regime_net=10.0,
        regime_er_hi=0.45, regime_er_lo=0.25, regime_rng=25.0,
        regime_pers=6),
    # same setup without the regime gate and without the Thursday rule
    "wick_fade": dict(
        stop_pips=10.0, be_trigger_pips=12.0, take_pips=32.0,
        be_offset_pips=0.0, exit_mode=0, partial_size=0.01,
        max_spread_pips=1.5, cycle=24, oversold=22.5, overbought=80.0,
        wick_frac=0.40, htf_filter=True, hour_from=14, hour_to=18,
        days_mask=127, thu_from=-1, compound=True, ladder_step=50.0,
        lot_step=0.01, max_lot=0.20, fixed_lot=0.01,
        regime_gate=False),
}


# ---------------------------------------------------------------------------
# signal generation — runs on closed bars, emits decision-time signals
# ---------------------------------------------------------------------------
def generate_signals(bars, start_ms, end_ms, cfg):
    t0, o, h, l, c = bars["t0"], bars["o"], bars["h"], bars["l"], bars["c"]
    n = len(t0)
    crsi_fast = CyclicRSI(cfg["cycle"])
    crsi_slow = CyclicRSI(cfg["cycle"] * 4)
    opinion = (regime_opinion(
                bars, K=cfg.get("regime_K", 20),
                net_min=cfg.get("regime_net", 15.0),
                er_hi=cfg.get("regime_er_hi", 0.40),
                er_lo=cfg.get("regime_er_lo", 0.28),
                rng_max=cfg.get("regime_rng", 30.0),
                pers=cfg.get("regime_pers", 3), pip=PIP)
            if cfg.get("regime_gate") else None)
    days = (t0 // 86_400_000).astype(np.int64)
    dow = (4 + days) % 7                     # 0=Sunday (1970-01-01 = Thursday = 4)
    hour = ((t0 // 3_600_000) % 24).astype(np.int64)

    signals = []
    for i in range(1, n - 1):
        change = c[i] - c[i - 1]
        crsi_f = crsi_fast.update(change)
        crsi_s = crsi_slow.update(change)
        if not (start_ms <= t0[i + 1] <= end_ms):
            continue
        if (cfg["days_mask"] & (1 << int(dow[i]))) == 0:
            continue
        if cfg["thu_from"] >= 0 and int(dow[i]) == 4 \
                and int(hour[i]) < cfg["thu_from"]:
            continue
        if int(hour[i]) < cfg["hour_from"] or int(hour[i]) > cfg["hour_to"]:
            continue
        if opinion is not None and opinion[i] in (2, 3):
            continue                               # regime gate: no fades in legs
        bar_range = h[i] - l[i]
        if bar_range < 3.0 * PIP:
            continue
        body_lo = min(o[i], c[i])
        body_hi = max(o[i], c[i])
        lower_wick = (body_lo - l[i]) / bar_range
        upper_wick = (h[i] - body_hi) / bar_range
        buy = crsi_f <= (cfg["oversold"] + 5.0) and lower_wick >= cfg["wick_frac"] \
            and (not cfg["htf_filter"] or crsi_s <= 40.0)
        sell = (not buy) and crsi_f >= (cfg["overbought"] - 5.0) and upper_wick >= cfg["wick_frac"] \
            and (not cfg["htf_filter"] or crsi_s >= 60.0)
        if buy or sell:
            signals.append({
                "ts": int(t0[i + 1]),               # next bar open = decision time
                "dir": "BUY" if buy else "SELL",
                "meta": {"signal_t": int(t0[i]), "crsi_f": crsi_f, "crsi_s": crsi_s},
            })
    return signals


# ---------------------------------------------------------------------------
# sizing — fixed lot or compound ladder via the engine's callable vol
# ---------------------------------------------------------------------------
def make_sizer(cfg):
    if not cfg["compound"]:
        return cfg["fixed_lot"]

    def sizer(ctx):
        equity = ctx["equity"]
        deposit0 = ctx["deposit"]
        if cfg.get("profit_ladder"):
            steps = int(max(0.0, equity - deposit0) // cfg["ladder_step"])
            return min(cfg["max_lot"], (steps + 1) * cfg["lot_step"])
        return max(cfg["lot_step"],
                   min(cfg["max_lot"],
                       (int(equity // cfg["ladder_step"])) * cfg["lot_step"]))

    return sizer


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def print_report(trades, label, date_from, date_to, deposit):
    if not trades:
        print(f"{label} {date_from}->{date_to}: 0 trades")
        return
    tp = sum(1 for p in trades if p.reason == "tp")
    sl = sum(1 for p in trades if p.reason == "sl")
    be = sum(1 for p in trades if p.reason == "sl" and abs(p.pnl_pips) < 0.6)
    gp = sum(p.pnl_usd for p in trades if p.pnl_usd > 0)
    gl = abs(sum(p.pnl_usd for p in trades if p.pnl_usd <= 0))
    nb = sum(1 for p in trades if p.direction == "BUY")
    pnl_total = sum(p.pnl_usd for p in trades)
    final_balance = deposit + pnl_total
    print(f"\n=== {label} {date_from}->{date_to} ===")
    print(f"trades={len(trades)} (BUY {nb} / SELL {len(trades) - nb}) | "
          f"pnl=${pnl_total:+.2f} | PF={gp / max(gl, 1e-9):.2f}")
    print(f"  tp={tp} sl={sl} (sl~BE={be}) | final balance=${final_balance:.2f}")
    print(f"  BUY ${sum(p.pnl_usd for p in trades if p.direction == 'BUY'):+.2f} | "
          f"SELL ${sum(p.pnl_usd for p in trades if p.direction == 'SELL'):+.2f}")
    monthly = {}
    for p in trades:
        ym = pd.Timestamp(p.exit_ts, unit="ms", tz="UTC").strftime("%Y-%m")
        monthly[ym] = monthly.get(ym, 0.0) + p.pnl_usd
    n_months = len(monthly)
    if n_months:
        print(f"  pnl/month=${pnl_total / n_months:+.2f} | pnl/year=${pnl_total / n_months * 12:+.2f} "
              f"({(pnl_total / n_months * 12 / deposit) * 100:+.1f}%/yr on ${deposit:.0f})")
        pos = sum(1 for v in monthly.values() if v > 0)
        print(f"  positive months: {pos}/{n_months}")


def export_csv(trades, path):
    df = pd.DataFrame([{
        "dir": t.direction,
        "ent_t": pd.Timestamp(t.entry_ts, unit="ms", tz="UTC").strftime("%Y.%m.%d %H:%M:%S"),
        "ext_t": pd.Timestamp(t.exit_ts, unit="ms", tz="UTC").strftime("%Y.%m.%d %H:%M:%S"),
        "entry": f"{t.entry:.5f}", "vol": t.volume, "usd": f"{t.pnl_usd:.2f}",
        "pips": f"{t.pnl_pips:.1f}", "reason": t.reason, "nights": t.nights,
        "swap_days": t.swap_days,
    } for t in trades])
    df.to_csv(path, sep=";", index=False)
    print(f"  trades -> {path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin-dir", required=True, help="folder with <SIDE>_<YEAR>.npz")
    ap.add_argument("--from", dest="date_from", required=True,
                    help="test start YYYY-MM-DD (UTC)")
    ap.add_argument("--to", dest="date_to", required=True,
                    help="test end YYYY-MM-DD (UTC)")
    ap.add_argument("--preset", default="wick_fade_regime", choices=list(PRESETS))
    ap.add_argument("--deposit", type=float, default=150.0)
    ap.add_argument("--tf", default="15m", help="bar size built from ticks")
    ap.add_argument("--commission", type=float, default=3.67,
                    help="round-trip commission USD per 1.0 lot (default: one "
                         "broker's raw-spread account; use yours)")
    ap.add_argument("--swap-buy", type=float, default=-7.08,
                    help="swap USD per 1.0 lot per swap day, BUY")
    ap.add_argument("--swap-sell", type=float, default=1.10,
                    help="swap USD per 1.0 lot per swap day, SELL")
    ap.add_argument("--out", default=None, help="trades CSV path")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    cfg.setdefault("ladder_step", 50.0)
    start_ms = int(dt.datetime.strptime(args.date_from, "%Y-%m-%d")
                .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    end_ms = int((dt.datetime.strptime(args.date_to, "%Y-%m-%d")
                  + dt.timedelta(days=1))
                 .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

    bar_ms = timeframe_ms(args.tf)
    # load only the window needed (plus warm-up / tail), not whole years
    bars_from = start_ms - WARMUP_DAYS * DAY_MS
    ticks_to = end_ms + TAIL_DAYS * DAY_MS

    # bars from BID ticks (same convention as MT4/MT5 charts)
    ts_b, px_b = load_ticks(args.bin_dir, "BID", start_ms=bars_from, end_ms=end_ms)
    bars = build_bars(ts_b, px_b, bar_ms)
    del ts_b, px_b
    print(f"bars {args.tf}: {len(bars['t0'])}")
    if len(bars["t0"]) == 0:
        raise SystemExit(f"no BID ticks in {args.bin_dir} for {args.date_from}..{args.date_to}")

    print("generating signals...", flush=True)
    signals = generate_signals(bars, start_ms, end_ms, cfg)
    print(f"{len(signals)} candidate signals")

    print("loading ticks...", flush=True)
    data = {}
    for side in ("BID", "ASK"):
        data[side] = load_ticks(args.bin_dir, side, start_ms=start_ms - DAY_MS,
                                end_ms=ticks_to)
        print(f"  {side}: {len(data[side][0]) / 1e6:.1f}M ticks")

    costs = CostModel(
        pip_size=PIP,
        pip_value_usd_per_lot=10.0,
        commission_per_lot_rt=args.commission,
        swap_per_lot_night_buy=args.swap_buy,
        swap_per_lot_night_sell=args.swap_sell,
        max_spread_pips=cfg["max_spread_pips"],
    )
    exits = ExitRules(
        stop_pips=cfg["stop_pips"], take_pips=cfg["take_pips"],
        be_trigger_pips=cfg["be_trigger_pips"],
        be_offset_pips=cfg["be_offset_pips"],
        partial_lots=cfg["partial_size"] if cfg["exit_mode"] == 1 else 0.0,
    )
    sizer = make_sizer(cfg)
    for s in signals:
        s["vol"] = sizer

    engine = ReplayEngine(data["BID"][0], data["BID"][1],
                          data["ASK"][0], data["ASK"][1], costs=costs)
    print("replaying...", flush=True)
    trades = engine.run(signals, exits, deposit=args.deposit)
    print_report(trades, args.preset, args.date_from, args.date_to, args.deposit)

    out = args.out or f"trades_{args.preset}_{args.date_from}_{args.date_to}.csv"
    export_csv(trades, out)


if __name__ == "__main__":
    main()
