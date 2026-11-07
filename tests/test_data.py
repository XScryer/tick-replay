import numpy as np
import pytest

from tickreplay.data import (
    _merge_side_csvs,
    _split_by_year,
    build_bars,
    convert_csvs_to_bin,
    load_ticks,
    timeframe_ms,
)

Y2024 = 1_704_067_200_000          # 2024-01-01 00:00 UTC


def _write_csv(path, ts, px):
    with open(path, "w") as f:
        f.write("time_ms,time,price\n")
        for t, p in zip(ts, px):
            f.write(f"{t},x,{p:.5f}\n")


def test_build_bars_ohlc():
    ts = np.array([0, 1, 2, 60_000, 60_500, 180_000], dtype=np.int64)
    px = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0])
    b = build_bars(ts, px, 60_000)
    assert list(b["t0"]) == [0, 60_000, 180_000]       # empty minute is absent
    assert list(b["o"]) == [1.0, 5.0, 7.0]
    assert list(b["h"]) == [3.0, 5.0, 7.0]
    assert list(b["l"]) == [1.0, 4.0, 7.0]
    assert list(b["c"]) == [2.0, 4.0, 7.0]
    assert list(b["t1"] - b["t0"]) == [60_000] * 3
    assert len(build_bars(ts[:0], px[:0], 60_000)["t0"]) == 0


def test_timeframe_ms():
    assert timeframe_ms("15m") == 900_000
    assert timeframe_ms("1h") == 3_600_000
    assert timeframe_ms("1d") == 86_400_000
    with pytest.raises(ValueError):
        timeframe_ms("5x")


def test_merge_sorts_and_drops_exact_duplicates(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_csv(a, [3, 1, 2], [1.3, 1.1, 1.2])
    _write_csv(b, [2, 2, 4], [1.2, 1.25, 1.4])          # (2, 1.2) duplicated
    ts, px = _merge_side_csvs([str(a), str(b)])
    assert list(ts) == [1, 2, 2, 3, 4]
    assert list(px) == [1.1, 1.2, 1.25, 1.3, 1.4]       # same ms, new price kept


def test_split_by_year_across_new_year():
    ts = np.array([Y2024 - 2, Y2024 - 1, Y2024, Y2024 + 5], dtype=np.int64)
    parts = list(_split_by_year(ts, ts.astype(float)))
    assert [(y, len(t)) for y, t, _ in parts] == [(2023, 2), (2024, 2)]


def test_convert_and_windowed_load(tmp_path):
    raw, bin_dir = tmp_path / "raw", tmp_path / "bin"
    raw.mkdir()
    ts = [Y2024 - 3_600_000, Y2024 + 1_000, Y2024 + 2_000, Y2024 + 86_400_000]
    _write_csv(raw / "EURUSD_ticks_x_y_BID.csv", ts, [1.1, 1.2, 1.3, 1.4])
    _write_csv(raw / "EURUSD_ticks_x_y_ASK.csv", ts, [1.2, 1.3, 1.4, 1.5])
    _write_csv(raw / "GBPUSD_ticks_x_y_BID.csv", [1], [9.9])   # other symbol
    convert_csvs_to_bin(str(raw), "EURUSD", str(bin_dir), log=lambda *_: None)
    assert sorted(p.name for p in bin_dir.iterdir()) == [
        "ASK_2023.npz", "ASK_2024.npz", "BID_2023.npz", "BID_2024.npz"]
    t, p = load_ticks(str(bin_dir), "BID")
    assert list(t) == ts and abs(p[0] - 1.1) < 1e-12
    t, _ = load_ticks(str(bin_dir), "BID", start_ms=Y2024, end_ms=Y2024 + 2_000)
    assert list(t) == [Y2024 + 1_000]                  # end is exclusive
    t, _ = load_ticks(str(bin_dir), "ASK", years=[2023])
    assert list(t) == [Y2024 - 3_600_000]
    t, p = load_ticks(str(tmp_path / "nope"), "BID")
    assert t.dtype == np.int64 and len(t) == 0
