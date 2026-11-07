"""Worked example: EURUSD M15 wick-fade strategy on real cTrader ticks.

This is a complete, runnable strategy — not a toy. It ports a real
research-grade setup:

  - CyclicRSI (cRSI) on bar closes, two speeds (fast + 4x slow for HTF filter)
  - Entry: rejection-wick fade — long lower wick / short upper wick
  - Session filter: only 14:00-18:00 UTC (NY morning), optional Thursday rule
  - Optional regime gate: skip fades while the market is in a confirmed leg
    (tickreplay.regime.regime_opinion)
  - Exits on ticks: SL / TP / break-even trigger, real spread filter,
    measured commission and swap
  - Position sizing: fixed lot or the compound "ladder" (volume grows with
    equity — implemented through the engine's callable `vol`)

Requires EURUSD tick data in .npz format (see README quickstart):

    py -m tickreplay.ctrader --symbol EURUSD --from 2019-01-01 --to 2026-12-31 --out data/EURUSD/raw
    py -m tickreplay.data csv2bin --raw-dir data/EURUSD/raw --symbol EURUSD --bin-dir data/EURUSD/bin
    py examples/eurusd_wick_fade.py --bin-dir data/EURUSD/bin --de 2020-01-01 --ate 2024-12-31
"""

import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd

from tickreplay.data import build_bars, load_ticks
from tickreplay.regime import regime_opinion
from tickreplay.replay import (CostModel, ExitRules, ReplayEngine)

PIP = 0.0001
BAR_15M_MS = 900_000


# ---------------------------------------------------------------------------
# CyclicRSI — port of the cRSI engine used by the strategy
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
# configuration — presets measured on EURUSD M15 real ticks
# ---------------------------------------------------------------------------
PRESETS = {
    # validated configuration (regime gate K=12, net=10p, ER>=0.45,
    # consensus 6 bars) — see repo docs for the research behind it
    "quilate_regK": dict(
        stop_pips=10.0, gatilho_pips=12.0, take_pips=32.0,
        be_offset_pips=0.0, modo_saida=0, vol_parcial=0.01,
        max_spread_pips=1.5, ciclo=24, os_thr=22.5, ob_thr=80.0,
        wick_pct=0.40, htf_on=True, hora_ini=14, hora_fim=18,
        dias_mask=127, qui_desde=16, composto=True, escada_lucro=False,
        degrau_saldo=50.0, lote_degrau=0.01, lote_max=0.20, lote_fixo=0.01,
        regime_gate=True, regime_K=12, regime_net=10.0,
        regime_er_hi=0.45, regime_er_lo=0.25, regime_rng=25.0,
        regime_pers=6),
    # same setup without the regime gate
    "campeao": dict(
        stop_pips=10.0, gatilho_pips=12.0, take_pips=32.0,
        be_offset_pips=0.0, modo_saida=0, vol_parcial=0.01,
        max_spread_pips=1.5, ciclo=24, os_thr=22.5, ob_thr=80.0,
        wick_pct=0.40, htf_on=True, hora_ini=14, hora_fim=18,
        dias_mask=127, qui_desde=-1, composto=True, degrau_saldo=50.0,
        lote_degrau=0.01, lote_max=0.20, lote_fixo=0.01,
        regime_gate=False),
}


def _tf_to_ms(tf):
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    num = "".join(ch for ch in tf if ch.isdigit())
    unit = "".join(ch for ch in tf if ch.isalpha()) or "m"
    return int(num or 1) * mult[unit[0]] * 1000


# ---------------------------------------------------------------------------
# signal generation — runs on closed bars, emits decision-time signals
# ---------------------------------------------------------------------------
def generate_signals(bars, de_ms, ate_ms, cfg):
    t0, o, h, l, c = bars["t0"], bars["o"], bars["h"], bars["l"], bars["c"]
    n = len(t0)
    crsi_l = CyclicRSI(cfg["ciclo"])
    crsi_h = CyclicRSI(cfg["ciclo"] * 4)
    opin = (regime_opinion(
                bars, K=cfg.get("regime_K", 20),
                net_min=cfg.get("regime_net", 15.0),
                er_hi=cfg.get("regime_er_hi", 0.40),
                er_lo=cfg.get("regime_er_lo", 0.28),
                rng_max=cfg.get("regime_rng", 30.0),
                pers=cfg.get("regime_pers", 3), pip=PIP)
            if cfg.get("regime_gate") else None)
    days = (t0 // 86_400_000).astype(np.int64)
    dow_mql = (4 + days) % 7                     # epoch = Thursday (MQL 4)
    hora = ((t0 // 3_600_000) % 24).astype(np.int64)

    signals = []
    for i in range(1, n - 1):
        chg = c[i] - c[i - 1]
        cl = crsi_l.update(chg)
        ch = crsi_h.update(chg)
        if not (de_ms <= t0[i + 1] <= ate_ms):
            continue
        if (cfg["dias_mask"] & (1 << int(dow_mql[i]))) == 0:
            continue
        if cfg["qui_desde"] >= 0 and int(dow_mql[i]) == 4 \
                and int(hora[i]) < cfg["qui_desde"]:
            continue
        if int(hora[i]) < cfg["hora_ini"] or int(hora[i]) > cfg["hora_fim"]:
            continue
        if opin is not None and opin[i] in (2, 3):
            continue                               # regime gate: no fades in legs
        rng = h[i] - l[i]
        if rng < 3.0 * PIP:
            continue
        body_lo = min(o[i], c[i])
        body_hi = max(o[i], c[i])
        lw = (body_lo - l[i]) / rng
        uw = (h[i] - body_hi) / rng
        buy = cl <= (cfg["os_thr"] + 5.0) and lw >= cfg["wick_pct"] \
            and (not cfg["htf_on"] or ch <= 40.0)
        sell = (not buy) and cl >= (cfg["ob_thr"] - 5.0) and uw >= cfg["wick_pct"] \
            and (not cfg["htf_on"] or ch >= 60.0)
        if buy or sell:
            signals.append({
                "ts": int(t0[i + 1]),               # next bar open = decision time
                "dir": "BUY" if buy else "SELL",
                "meta": {"sig_t": int(t0[i]), "cl": cl, "ch": ch},
            })
    return signals


# ---------------------------------------------------------------------------
# sizing — fixed lot or compound ladder via the engine's callable vol
# ---------------------------------------------------------------------------
def make_sizer(cfg):
    if not cfg["composto"]:
        return cfg["lote_fixo"]

    def sizer(ctx):
        equity = ctx["equity"]
        dep0 = ctx["deposit"]
        if cfg.get("escada_lucro"):
            degraus = int(max(0.0, equity - dep0) // cfg["degrau_saldo"])
            return min(cfg["lote_max"], (degraus + 1) * cfg["lote_degrau"])
        return max(cfg["lote_degrau"],
                   min(cfg["lote_max"],
                       (int(equity // cfg["degrau_saldo"])) * cfg["lote_degrau"]))

    return sizer


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def print_report(trades, label, de, ate, deposit):
    if not trades:
        print(f"{label} {de}->{ate}: 0 trades")
        return
    tp = sum(1 for p in trades if p.reason == "tp")
    sl = sum(1 for p in trades if p.reason == "sl")
    be = sum(1 for p in trades if p.reason == "sl" and abs(p.pnl_pips) < 0.6)
    gp = sum(p.pnl_usd for p in trades if p.pnl_usd > 0)
    gl = abs(sum(p.pnl_usd for p in trades if p.pnl_usd <= 0))
    nb = sum(1 for p in trades if p.direction == "BUY")
    lucro = sum(p.pnl_usd for p in trades)
    saldo_fim = deposit + lucro
    print(f"\n=== {label} {de}->{ate} ===")
    print(f"trades={len(trades)} (BUY {nb} / SELL {len(trades) - nb}) | "
          f"pnl=${lucro:+.2f} | PF={gp / max(gl, 1e-9):.2f}")
    print(f"  tp={tp} sl={sl} (sl~BE={be}) | final balance=${saldo_fim:.2f}")
    print(f"  BUY ${sum(p.pnl_usd for p in trades if p.direction == 'BUY'):+.2f} | "
          f"SELL ${sum(p.pnl_usd for p in trades if p.direction == 'SELL'):+.2f}")
    mensal = {}
    for p in trades:
        ym = pd.Timestamp(p.exit_ts, unit="ms", tz="UTC").strftime("%Y-%m")
        mensal[ym] = mensal.get(ym, 0.0) + p.pnl_usd
    nmeses = len(mensal)
    if nmeses:
        print(f"  pnl/month=${lucro / nmeses:+.2f} | pnl/year=${lucro / nmeses * 12:+.2f} "
              f"({(lucro / nmeses * 12 / deposit) * 100:+.1f}%/yr on ${deposit:.0f})")
        pos = sum(1 for v in mensal.values() if v > 0)
        print(f"  positive months: {pos}/{nmeses}")


def export_csv(trades, path):
    df = pd.DataFrame([{
        "dir": t.direction,
        "ent_t": pd.Timestamp(t.entry_ts, unit="ms", tz="UTC").strftime("%Y.%m.%d %H:%M:%S"),
        "ext_t": pd.Timestamp(t.exit_ts, unit="ms", tz="UTC").strftime("%Y.%m.%d %H:%M:%S"),
        "entry": f"{t.entry:.5f}", "vol": t.volume, "usd": f"{t.pnl_usd:.2f}",
        "pips": f"{t.pnl_pips:.1f}", "reason": t.reason, "nights": t.nights,
    } for t in trades])
    df.to_csv(path, sep=";", index=False)
    print(f"  trades -> {path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin-dir", required=True, help="folder with <SIDE>_<YEAR>.npz")
    ap.add_argument("--de", required=True, help="test start YYYY-MM-DD (UTC)")
    ap.add_argument("--ate", required=True, help="test end YYYY-MM-DD (UTC)")
    ap.add_argument("--preset", default="quilate_regK", choices=list(PRESETS))
    ap.add_argument("--deposit", type=float, default=150.0)
    ap.add_argument("--tf", default="15m", help="bar size built from ticks")
    ap.add_argument("--commission", type=float, default=3.67,
                    help="round-trip commission USD per 1.0 lot")
    ap.add_argument("--swap-buy", type=float, default=-7.08,
                    help="swap USD per 1.0 lot per night, BUY")
    ap.add_argument("--swap-sell", type=float, default=1.10,
                    help="swap USD per 1.0 lot per night, SELL")
    ap.add_argument("--out", default=None, help="trades CSV path")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    cfg.setdefault("degrau_saldo", 50.0)
    de_ms = int(dt.datetime.strptime(args.de, "%Y-%m-%d")
                .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    ate_ms = int((dt.datetime.strptime(args.ate, "%Y-%m-%d")
                  + dt.timedelta(days=1))
                 .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

    bar_ms = _tf_to_ms(args.tf)
    a_ini = pd.Timestamp(de_ms, unit="ms", tz="UTC").year - 1
    a_fim = pd.Timestamp(ate_ms, unit="ms", tz="UTC").year
    anos_barra = list(range(a_ini, a_fim + 1))
    anos_tick = list(range(a_ini - 1, a_fim + 2))
    print(f"bars {args.tf}: years {anos_barra} | ticks: years {anos_tick}")

    # bars from BID ticks (broker convention: bars print on bid)
    ts_b, px_b = load_ticks(args.bin_dir, "BID", anos_barra)
    bars = build_bars(ts_b, px_b, bar_ms)
    print(f"bars: {len(bars['t0'])}")

    print("generating signals...", flush=True)
    signals = generate_signals(bars, de_ms, ate_ms, cfg)
    print(f"{len(signals)} candidate signals")

    print("loading ticks...", flush=True)
    data = {}
    for side in ("BID", "ASK"):
        ts, px = load_ticks(args.bin_dir, side, anos_tick)
        lo = de_ms - 2 * 86_400_000
        keep = ts >= lo
        data[side] = (ts[keep], px[keep])
        print(f"  {side}: {len(ts[keep]) / 1e6:.1f}M ticks")

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
        be_trigger_pips=cfg["gatilho_pips"],
        be_offset_pips=cfg["be_offset_pips"],
        partial_lots=cfg["vol_parcial"] if cfg["modo_saida"] == 1 else 0.0,
    )
    sizer = make_sizer(cfg)
    for s in signals:
        s["vol"] = sizer

    engine = ReplayEngine(data["BID"][0], data["BID"][1],
                          data["ASK"][0], data["ASK"][1], costs=costs)
    print("replaying...", flush=True)
    trades = engine.run(signals, exits, deposit=args.deposit)
    print_report(trades, args.preset, args.de, args.ate, args.deposit)

    out = args.out or f"trades_{args.preset}_{args.de}_{args.ate}.csv"
    export_csv(trades, out)


if __name__ == "__main__":
    main()
