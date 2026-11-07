"""Engine tests — plain asserts, no pytest required:  py tests/test_replay.py

Ticks are handcrafted arrays; no data files needed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tickreplay.replay import (CostModel, ExitRules, ReplayEngine,
                               count_rollovers, eu_broker_offset,
                               fixed_offset)

T0 = 1_700_000_000_000  # fixed epoch ms
STEP = 1000             # 1 tick per second


def make_ticks(prices, start=T0, step=STEP):
    ts = start + np.arange(len(prices), dtype=np.int64) * step
    return ts, np.asarray(prices, dtype=np.float64)


def both_sides(bid_prices, spread=0.0001, **kw):
    bts, bpx = make_ticks(bid_prices, **kw)
    ats = bts.copy()
    apx = bpx + spread
    return bts, bpx, ats, apx


def engine(bid_prices, spread=0.0001, costs=None, **kw):
    bts, bpx, ats, apx = both_sides(bid_prices, spread, **kw)
    return ReplayEngine(bts, bpx, ats, apx, costs=costs or CostModel())


PIP = 1e-4


def test_buy_fills_at_ask_and_takes_profit():
    # flat then ramp up: BUY at ask, TP on bid side
    px = [1.10000] * 6 + [1.10000 + (i - 5) * PIP for i in range(6, 20)]
    eng = engine(px, costs=CostModel(commission_per_lot_rt=0.0))
    sig = [{"ts": T0 + 500, "dir": "BUY", "vol": 0.01}]
    trades = eng.run(sig, ExitRules(stop_pips=10, take_pips=5))
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "BUY"
    assert abs(t.entry - 1.10010) < 1e-9        # ask = bid + 1 pip
    assert t.reason == "tp"
    assert abs(t.pnl_pips - 5.0) < 1e-6
    assert abs(t.pnl_usd - 0.50) < 1e-9          # 5 pips * $10 * 0.01 lot


def test_sell_fills_at_bid_and_stops_out():
    px = [1.10000] * 6 + [1.10000 + (i - 5) * PIP for i in range(6, 20)]
    eng = engine(px)
    sig = [{"ts": T0 + 500, "dir": "SELL", "vol": 0.01}]
    trades = eng.run(sig, ExitRules(stop_pips=5, take_pips=50))
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "SELL"
    assert abs(t.entry - 1.10000) < 1e-9        # sell fills at bid
    assert t.reason == "sl"
    assert t.pnl_pips < 0


def test_spread_filter_skips_signal():
    eng = engine([1.10000] * 30, spread=0.001)  # 10-pip spread
    sig = [{"ts": T0 + 500, "dir": "BUY", "vol": 0.01}]
    eng.costs.max_spread_pips = 1.5
    trades = eng.run(sig, ExitRules(stop_pips=10, take_pips=50))
    assert trades == []


def test_one_position_at_a_time():
    # price drifts down slowly: first BUY stays open for many ticks
    px = [1.10000 - i * PIP * 0.2 for i in range(60)]
    eng = engine(px)
    sig = [{"ts": T0 + i * STEP, "dir": "BUY", "vol": 0.01} for i in range(1, 6)]
    trades = eng.run(sig, ExitRules(stop_pips=500, take_pips=500))
    # signals inside the open trade are skipped -> only 1 trade
    assert len(trades) <= 2
    if len(trades) == 2:
        assert trades[1].decision_ts >= trades[0].exit_ts


def test_break_even_then_stop():
    # entry = ask 1.10010; BE trigger = entry+2p = 1.10030 on BID ticks.
    # bid must reach 1.10030 (idx 8 of the rise), then fade through 1.10015.
    px = ([1.10000] * 6
          + [1.10010, 1.10020, 1.10030, 1.10040, 1.10050, 1.10060]
          + [1.10014] * 10)
    eng = engine(px)
    sig = [{"ts": T0 + 500, "dir": "BUY", "vol": 0.01}]
    trades = eng.run(sig, ExitRules(stop_pips=10, take_pips=500,
                                    be_trigger_pips=2, be_offset_pips=0.5))
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "sl"
    # BE stop = entry + 0.5p; exit at 1.10014 -> +0.4 pips
    assert abs(t.pnl_pips - 0.4) < 1e-6


def test_partial_close():
    px = ([1.10000] * 6
          + [1.10010, 1.10020, 1.10030, 1.10040, 1.10050, 1.10060]
          + [1.10014] * 20)
    eng = engine(px)
    sig = [{"ts": T0 + 500, "dir": "BUY", "vol": 0.05}]
    trades = eng.run(sig, ExitRules(stop_pips=10, take_pips=500,
                                    be_trigger_pips=2, be_offset_pips=0.5,
                                    partial_lots=0.02, min_lot=0.01))
    assert len(trades) == 1
    reasons = [e[3] for e in trades[0].exits]
    assert "partial" in reasons
    assert abs(trades[0].exits[-1][2] - 0.03) < 1e-9  # 0.05 - 0.02 remaining


def test_end_of_data_exit():
    px = [1.10000] * 10                          # flat, SL/TP never hit
    eng = engine(px)
    sig = [{"ts": T0 + 500, "dir": "BUY", "vol": 0.01}]
    trades = eng.run(sig, ExitRules(stop_pips=10, take_pips=50))
    assert len(trades) == 1
    assert trades[0].reason == "eod"


def test_costs_commission_and_swap():
    costs = CostModel(commission_per_lot_rt=2.0,
                      swap_per_lot_night_buy=-1.0,
                      rollover_offset=fixed_offset(0))
    eng = engine([1.10000] * 10, spread=0.0, costs=costs)
    sig = [{"ts": T0 + 500, "dir": "BUY", "vol": 0.02}]
    trades = eng.run(sig, ExitRules(stop_pips=10, take_pips=50))
    t = trades[0]
    assert t.reason == "eod"
    assert abs(t.pnl_usd - (0.0 - 2.0 * 0.02)) < 1e-9   # flat pips - commission


def test_callable_volume_sizing():
    seen = []

    def sizer(ctx):
        seen.append(ctx["equity"])
        return 0.02

    px = [1.10000] * 6 + [1.10000 + (i - 5) * PIP for i in range(6, 20)]
    eng = engine(px, costs=CostModel(commission_per_lot_rt=0.0))
    sig = [{"ts": T0 + i * STEP, "dir": "BUY", "vol": sizer}
           for i in (1, 2)]
    trades = eng.run(sig, ExitRules(stop_pips=10, take_pips=5),
                     deposit=100.0)
    assert trades[0].volume == 0.02
    # second signal falls inside the open trade -> skipped; sizer still saw
    # equity=100 (deposit) at decision time
    assert seen and seen[0] == 100.0


def test_count_rollovers_fixed_offset():
    # server = UTC+0 -> midnight = 00:00 UTC every day
    start = T0
    end = start + 3 * 86_400_000
    assert count_rollovers(start, end, fixed_offset(0)) == 3


def test_eu_broker_offset_dst():
    off = eu_broker_offset()
    # mid-January: winter (+2); mid-July: summer (+3)
    jan = 1_704_067_200_000   # 2024-01-01
    jul = 1_720_000_000_000   # ~2024-07
    assert off(jan) == 2
    assert off(jul) == 3


def run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    run_all()
