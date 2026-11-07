"""Engine tests. Ticks are handcrafted arrays; no data files needed."""

import datetime as dt

import numpy as np

from tickreplay.replay import (
    CostModel,
    ExitRules,
    ReplayEngine,
    count_rollovers,
    eu_broker_offset,
    fixed_offset,
    ny_close_offset,
    rollover_times,
    swap_days,
)

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
    # every later signal falls inside the open trade -> skipped
    assert len(trades) == 1
    assert trades[0].reason == "eod"


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


def test_costs_commission_without_rollover():
    costs = CostModel(commission_per_lot_rt=2.0,
                      swap_per_lot_night_buy=-1.0,
                      rollover_offset=fixed_offset(0))
    eng = engine([1.10000] * 10, spread=0.0, costs=costs)
    sig = [{"ts": T0 + 500, "dir": "BUY", "vol": 0.02}]
    trades = eng.run(sig, ExitRules(stop_pips=10, take_pips=50))
    t = trades[0]
    assert t.reason == "eod" and t.nights == 0
    assert abs(t.pnl_usd - (0.0 - 2.0 * 0.02)) < 1e-9   # flat pips - commission


def _ms(s):
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def test_swap_charged_when_holding_over_rollover():
    # Wednesday 2024-01-10 20:00 UTC -> Thursday 10:00 UTC crosses the
    # Wednesday rollover (22:00 UTC in winter) -> triple swap
    start = _ms("2024-01-10T20:00:00")
    hours = 14
    px = [1.10000] * (hours + 1)
    costs = CostModel(swap_per_lot_night_buy=-2.0, swap_per_lot_night_sell=0.5)
    eng = engine(px, spread=0.0, costs=costs, start=start, step=3_600_000)
    for direction, per_day in (("BUY", -2.0), ("SELL", 0.5)):
        t = eng.run([{"ts": start, "dir": direction, "vol": 0.10}],
                    ExitRules(stop_pips=10, take_pips=50))[0]
        assert t.nights == 1 and t.swap_days == 3
        assert abs(t.pnl_usd - per_day * 0.10 * 3) < 1e-9


def test_swap_days_weekend_and_triple_wednesday():
    off = ny_close_offset()
    # Friday evening -> Monday: 3 server midnights, only Friday's is charged
    fri, mon = _ms("2024-01-05T20:00:00"), _ms("2024-01-08T10:00:00")
    assert count_rollovers(fri, mon, off) == 3
    assert swap_days(fri, mon, off) == 1
    # Wednesday -> Thursday: triple
    assert swap_days(_ms("2024-01-10T20:00:00"), _ms("2024-01-11T10:00:00"), off) == 3
    # a full week Monday -> Monday: 1+1+3+1+1+0+0
    assert swap_days(_ms("2024-01-08T10:00:00"), _ms("2024-01-15T10:00:00"), off) == 7
    # 7-day markets: every rollover once
    assert swap_days(fri, mon, off, triple_weekday=None) == 3


def test_ny_close_rollover_follows_us_dst():
    off = ny_close_offset()
    assert off(_ms("2024-01-15T12:00:00")) == 2
    assert off(_ms("2024-07-15T12:00:00")) == 3
    # US DST starts 2024-03-10 (EU only on 03-31): rollover moves 22:00 -> 21:00 UTC
    got = rollover_times(_ms("2024-03-08T12:00:00"), _ms("2024-03-12T12:00:00"), off)
    hours = [dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).hour for t in got]
    assert hours == [22, 22, 21, 21]


def test_decision_quote_includes_tick_at_decision_instant():
    # bid/ask jump exactly at the first tick after the decision time
    bts = np.array([T0, T0 + 1000, T0 + 2000, T0 + 3000], dtype=np.int64)
    bpx = np.array([1.10000, 1.10050, 1.10050, 1.10050])
    eng = ReplayEngine(bts, bpx, bts.copy(), bpx + 0.0001)
    t = eng.run([{"ts": T0 + 500, "dir": "BUY", "vol": 0.01}],
                ExitRules(stop_pips=10, take_pips=50))[0]
    assert t.entry_ts == T0 + 1000 and t.decision_ts == T0 + 500
    assert abs(t.entry - 1.10060) < 1e-9        # new ask, not the stale one


def test_gapped_stop_fills_at_worse_price():
    px = [1.10000] * 3 + [1.09800] + [1.09800] * 3   # 20-pip gap down
    eng = engine(px, spread=0.0)
    t = eng.run([{"ts": T0 + 500, "dir": "BUY", "vol": 0.01}],
                ExitRules(stop_pips=5, take_pips=50))[0]
    assert t.reason == "sl"
    assert abs(t.pnl_pips + 20.0) < 1e-6


def test_sell_take_profit():
    px = [1.10000] * 3 + [1.10000 - i * PIP for i in range(1, 12)]
    eng = engine(px)
    t = eng.run([{"ts": T0 + 500, "dir": "SELL", "vol": 0.01}],
                ExitRules(stop_pips=10, take_pips=5))[0]
    assert t.reason == "tp" and abs(t.pnl_pips - 5.0) < 1e-6


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
