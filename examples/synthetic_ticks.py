"""Generate synthetic tick data to try the pipeline without a broker account.

Writes CSVs in exactly the downloader's format
(<symbol>_ticks_<from>_<to>_<SIDE>.csv, columns time_ms,time,price) so the
next step is the real one:

    python examples/synthetic_ticks.py --days 90 --out data/SYNTH/raw
    python -m tickreplay.data csv2bin --raw-dir data/SYNTH/raw --symbol SYNTH --bin-dir data/SYNTH/bin
    python examples/eurusd_wick_fade.py --bin-dir data/SYNTH/bin \
        --from 2024-01-15 --to 2024-04-15 --preset wick_fade

The price is a random walk with volatility clustering and a realistic
~1-pip spread. It is NOT market data — it exists to smoke-test the engine.
"""

import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd

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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default="SYNTH")
    ap.add_argument("--days", type=int, default=90, help="trading days to generate")
    ap.add_argument("--ticks-per-day", type=int, default=60_000)
    ap.add_argument("--spread-pips", type=float, default=1.0)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--out", required=True, help="raw CSV dir")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    d0 = dt.datetime.strptime(args.start, "%Y-%m-%d")

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
        bid_rows.append((ts, bid))
        ask_rows.append((ts, ask))
        written_days += 1
        d += dt.timedelta(days=1)

    date_to = (d - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    for side, rows in (("BID", bid_rows), ("ASK", ask_rows)):
        ts = np.concatenate([r[0] for r in rows])
        px = np.concatenate([r[1] for r in rows])
        stamp = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y.%m.%d %H:%M:%S.%f").str[:-3]
        path = os.path.join(
            args.out, f"{args.symbol}_ticks_{args.start}_{date_to}_{side}.csv")
        pd.DataFrame({"time_ms": ts, "time": stamp, "price": px}).to_csv(
            path, index=False, float_format="%.5f")
        print(f"wrote {path}: {len(ts)} ticks")

    bin_dir = os.path.join(os.path.dirname(os.path.normpath(args.out)), "bin")
    print("\nNext:")
    print(f"  python -m tickreplay.data csv2bin --raw-dir {args.out} "
          f"--symbol {args.symbol} --bin-dir {bin_dir}")


if __name__ == "__main__":
    main()
