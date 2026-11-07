"""Tick data storage, loading, bar building and coverage audit.

On-disk layout (all paths are per-symbol directories):

    raw/                      CSVs straight from the downloader
        <SYMBOL>_ticks_<from>_<to>_<SIDE>.csv     columns: time_ms,time,price
        _progress.txt                              resume checkpoint (date + side)

    bin/                      compact per-year arrays, one file per side per year
        <SIDE>_<YEAR>.npz                          arrays: ts (int64 ms UTC), px (float64)

Bars are always derived from ticks (bid side) so the whole pipeline depends on
a single data source and stays internally consistent.
"""

import os
import numpy as np
import pandas as pd

SIDES = ("BID", "ASK")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def npz_files(bin_dir, side, years=None):
    """Sorted list of <SIDE>_<YEAR>.npz paths inside bin_dir."""
    side = side.upper()
    if not os.path.isdir(bin_dir):
        return []
    out = []
    for fn in sorted(os.listdir(bin_dir)):
        if not (fn.startswith(f"{side}_") and fn.endswith(".npz")):
            continue
        try:
            year = int(fn[len(side) + 1:-4])
        except ValueError:
            continue
        if years is not None and year not in years:
            continue
        out.append(os.path.join(bin_dir, fn))
    return out


def load_ticks(bin_dir, side, years=None):
    """Return (ts, px): int64 epoch-ms UTC and float64 prices, concatenated."""
    paths = npz_files(bin_dir, side, years)
    if not paths:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.float64))
    ts_parts, px_parts = [], []
    for p in paths:
        z = np.load(p)
        ts_parts.append(z["ts"])
        px_parts.append(z["px"])
    return np.concatenate(ts_parts), np.concatenate(px_parts)


def load_both_sides(bin_dir, years=None):
    """Return dict {'BID': (ts,px), 'ASK': (ts,px)}."""
    return {s: load_ticks(bin_dir, s, years) for s in SIDES}


# ---------------------------------------------------------------------------
# bars from ticks
# ---------------------------------------------------------------------------
def build_bars(ts, px, bar_ms):
    """Aggregate a tick series into OHLC bars of `bar_ms` milliseconds.

    Returns dict with t0 (bar open, UTC ms), t1 (bar close = t0+bar_ms),
    o/h/l/c. Bars with no ticks simply do not exist in the output.
    """
    n = len(ts)
    empty = {k: np.array([], dtype=np.float64) for k in ("t0", "t1", "o", "h", "l", "c")}
    if n == 0:
        return empty
    bar_id = ts // bar_ms
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = bar_id[1:] != bar_id[:-1]
    starts = np.nonzero(changed)[0]
    ends = np.append(starts[1:], n)
    t0 = (bar_id[starts] * bar_ms).astype(np.float64)
    return {
        "t0": t0,
        "t1": t0 + bar_ms,
        "o": px[starts].astype(np.float64),
        "h": np.maximum.reduceat(px, starts).astype(np.float64),
        "l": np.minimum.reduceat(px, starts).astype(np.float64),
        "c": px[ends - 1].astype(np.float64),
    }


# ---------------------------------------------------------------------------
# CSV -> npz conversion
# ---------------------------------------------------------------------------
def _iter_csv_ticks(path, chunksize=10_000_000):
    """Yield (ts int64, px float64) chunks from a downloader CSV."""
    for ch in pd.read_csv(path, usecols=[0, 2], names=["ts", "px"], header=0,
                          dtype={"ts": np.int64, "px": np.float64},
                          chunksize=chunksize):
        yield ch["ts"].to_numpy(), ch["px"].to_numpy()


def _merge_side_csvs(paths):
    """Concatenate CSVs, sort by timestamp and drop exact duplicates."""
    ts_parts, px_parts = [], []
    for path in paths:
        for ts, px in _iter_csv_ticks(path):
            ts_parts.append(ts)
            px_parts.append(px)
    if not ts_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    ts = np.concatenate(ts_parts)
    px = np.concatenate(px_parts)
    order = np.argsort(ts, kind="stable")
    ts, px = ts[order], px[order]
    if len(ts) > 1:
        dup = (ts[1:] == ts[:-1]) & (px[1:] == px[:-1])
        keep = np.ones(len(ts), dtype=bool)
        keep[1:][dup] = False
        ts, px = ts[keep], px[keep]
    return ts, px


def _split_by_year(ts, px):
    """Yield (year, ts_seg, px_seg) for contiguous year runs (ts must be sorted)."""
    years = (ts // 1000).astype("datetime64[s]").astype("datetime64[Y]").astype(np.int64) + 1970
    ini = 0
    for a in np.unique(years):
        idx = np.nonzero(years == a)[0]
        if len(idx) == 0:
            continue
        yield int(a), ts[ini:idx[-1] + 1], px[ini:idx[-1] + 1]
        ini = idx[-1] + 1


def convert_csvs_to_bin(raw_dir, symbol, bin_dir, sides=SIDES, log=print):
    """Merge every '<symbol>_ticks_*_<side>.csv' in raw_dir into per-year .npz.

    Safe to re-run: existing years are overwritten with the merged result.
    """
    os.makedirs(bin_dir, exist_ok=True)
    for side in sides:
        side = side.upper()
        paths = [os.path.join(raw_dir, fn) for fn in sorted(os.listdir(raw_dir))
                 if fn.startswith(f"{symbol}_") and fn.endswith(f"_{side}.csv")]
        if not paths:
            log(f"{side}: no CSVs found in {raw_dir}")
            continue
        log(f"{side}: {len(paths)} file(s)")
        ts, px = _merge_side_csvs(paths)
        log(f"{side}: {len(ts) / 1e6:.1f}M unique ticks")
        for year, seg_ts, seg_px in _split_by_year(ts, px):
            out = os.path.join(bin_dir, f"{side}_{year}.npz")
            np.savez_compressed(out, ts=seg_ts, px=seg_px)
            log(f"  wrote {out}: {len(seg_ts) / 1e6:.1f}M ticks")


# ---------------------------------------------------------------------------
# coverage audit
# ---------------------------------------------------------------------------
def audit_coverage(bin_dir, sides=SIDES, log=print):
    """Print per-year coverage: first/last tick, count, weekdays without ticks."""
    for side in sides:
        side = side.upper()
        log(f"\n===== {side} =====")
        for path in npz_files(bin_dir, side):
            year = int(os.path.basename(path)[len(side) + 1:-4])
            z = np.load(path)
            ts = z["ts"]
            if len(ts) == 0:
                log(f"{year}: EMPTY")
                continue
            d = pd.to_datetime(ts, unit="ms", utc=True)
            days = pd.Series(d.date).value_counts()
            idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B")
            missing = [x.date() for x in idx if x.date() not in days.index]
            monthly = d.strftime("%Y-%m").value_counts().sort_index()
            monthly_str = " ".join(f"{m[5:]}:{v / 1000:.0f}k" for m, v in monthly.items())
            log(f"{year}: {len(ts) / 1e6:7.1f}M ticks | {d[0]} -> {d[-1]} "
                f"| {len(days)} days with ticks | {len(missing)} weekdays without")
            log(f"    monthly(k): {monthly_str}")
            if 0 < len(missing) <= 30:
                log(f"    no ticks: {[str(x) for x in missing]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser(prog="tickreplay.data",
                                 description="Tick data utilities: csv->npz, bars, audit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("csv2bin", help="merge downloader CSVs into per-year .npz")
    p.add_argument("--raw-dir", required=True, help="folder with *_BID.csv / *_ASK.csv")
    p.add_argument("--symbol", required=True, help="symbol prefix, e.g. EURUSD")
    p.add_argument("--bin-dir", required=True, help="output folder for .npz files")

    p = sub.add_parser("bars", help="build OHLC bars from ticks and save as .npz")
    p.add_argument("--bin-dir", required=True)
    p.add_argument("--side", default="BID", choices=SIDES)
    p.add_argument("--tf", default="15m", help="bar size: 1m, 5m, 15m, 1h, 4h, 1d ...")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--years", nargs="*", type=int, default=None)

    p = sub.add_parser("audit", help="coverage report for a bin directory")
    p.add_argument("--bin-dir", required=True)
    p.add_argument("--sides", nargs="*", default=list(SIDES), choices=list(SIDES))

    args = ap.parse_args()
    if args.cmd == "csv2bin":
        convert_csvs_to_bin(args.raw_dir, args.symbol, args.bin_dir)
    elif args.cmd == "bars":
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        num = "".join(ch for ch in args.tf if ch.isdigit())
        unit = "".join(ch for ch in args.tf if ch.isalpha()) or "m"
        bar_ms = int(num or 1) * mult[unit[0]] * 1000
        ts, px = load_ticks(args.bin_dir, args.side, args.years)
        bars = build_bars(ts, px, bar_ms)
        np.savez_compressed(args.out, **bars)
        print(f"wrote {args.out}: {len(bars['t0'])} bars of {args.tf} "
              f"from {len(ts) / 1e6:.1f}M ticks")
    elif args.cmd == "audit":
        audit_coverage(args.bin_dir, sides=args.sides)


if __name__ == "__main__":
    _cli()
