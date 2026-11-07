"""Generate synthetic tick data to try the pipeline without a broker account.

Writes CSVs in exactly the downloader's format
(<symbol>_ticks_<from>_<to>_<SIDE>.csv, columns time_ms,time,price) so the
next step is the real one:

    py examples/synthetic_ticks.py --days 10 --out data/SYNTH/raw
    py -m tickreplay.data csv2bin --raw-dir data/SYNTH/raw --symbol SYNTH --bin-dir data/SYNTH/bin
    py examples/eurusd_wick_fade.py --bin-dir data/SYNTH/bin --de <d> --ate <d>

The price is a random walk with volatility clustering and a realistic
~1-pip spread. It is NOT market data — it exists to smoke-test the engine.
"""

import argparse
import datetime as dt
import os

import numpy as np

PIP = 0.0001


def gen_day(rng, day_start_ms, n_ticks, mid0, spread_pips):
    """One day of ticks: irregular timestamps + random-walk mid + spread."""
    # irregular inter-tick gaps: mostly fast, occasional pauses
    gaps = rng.exponential(700.0, n_ticks) + rng.uniform(0, 1500.0, n_ticks)
    ts = day_start_ms + np.cumsum(gaps).astype(np.int64)
    ts = ts[ts < day_start_ms + 86_400_000]
    n = len(ts)
    # volatility clustering: regimes of calm/turbulent
    regime = rng.choice([0.6, 1.0, 2.2], size=(n // 500) + 2)
    vol = np.repeat(regime, 500)[:n] * PIP * 0.05
    shocks = rng.normal(0, 1, n) * vol
    mid = mid0 + np.cumsum(shocks)
    spread = (spread_pips + rng.normal(0, 0.15, n).clip(-0.3, 0.8)) * PIP
    bid = mid - spread / 2
    ask = mid + spread / 2
    return ts, bid, ask, mid[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SYNTH")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--ticks-per-day", type=int, default=120_000)
    ap.add_argument("--spread-pips", type=float, default=1.0)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--out", required=True, help="raw CSV dir")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    d0 = dt.datetime.strptime(args.start, "%Y-%m-%d")
    de_str = args.start

    bid_rows, ask_rows = [], []
    d = d0
    written_days = 0
    mid = 1.1000
    while written_days < args.days:
        if d.weekday() == 5:          # skip Saturdays like forex does
            d += dt.timedelta(days=1)
            continue
        day_ms = int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        ts, bid, ask, mid = gen_day(rng, day_ms, args.ticks_per_day, mid,
                                    args.spread_pips)
        bid_rows.append(np.column_stack([ts, bid]))
        ask_rows.append(np.column_stack([ts, ask]))
        written_days += 1
        d += dt.timedelta(days=1)

    ate = (d - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    for side, rows in (("BID", bid_rows), ("ASK", ask_rows)):
        arr = np.vstack(rows)
        order = np.argsort(arr[:, 0], kind="stable")
        arr = arr[order]
        path = os.path.join(
            args.out, f"{args.symbol}_ticks_{de_str}_{ate}_{side}.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("time_ms,time,price\n")
            for t, px in arr:
                ds = dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime(
                    "%Y.%m.%d %H:%M:%S.%f")[:-3]
                f.write(f"{int(t)},{ds},{px:.5f}\n")
        print(f"wrote {path}: {len(arr)} ticks")

    print(f"\nNext:")
    print(f"  py -m tickreplay.data csv2bin --raw-dir {args.out} "
          f"--symbol {args.symbol} --bin-dir {args.out.replace('raw', 'bin')}")


if __name__ == "__main__":
    main()
