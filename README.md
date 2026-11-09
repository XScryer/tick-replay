# Tick Replay

[![CI](https://github.com/XScryer/tick-replay/actions/workflows/ci.yml/badge.svg)](https://github.com/XScryer/tick-replay/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Real bid/ask tick history from the cTrader Open API, plus a small,
explicit engine that replays trades tick by tick.**

Most free backtesters run on bars or on ticks *interpolated* from bars, so
fills and spreads are approximations. cTrader's Open API serves the actual
historical bid and ask quotes of the broker behind your cTrader ID — on a
free demo account too. This repo downloads them and replays your signals on
them with real spread, commission and swap.

```
cTrader Open API ──► raw CSVs ──► per-year .npz ──► bars / signals ──► tick replay ──► trades
  (demo or live)     (resumable)   (fast load)      (your strategy)    (bid/ask fills)  (report)
```

## What you get

- **A downloader** (`tickreplay.ctrader`) that handles OAuth (with token
  refresh), demo vs. live hosts, delta-encoded newest-first pagination and
  per-day checkpoints, so a multi-year download can be interrupted and
  resumed with the same command.
- **A data layer** (`tickreplay.data`) that merges CSVs into compact
  per-year `.npz` files, builds OHLC bars from ticks and reports coverage
  gaps.
- **A replay engine** (`tickreplay.replay`) that is strategy-agnostic: you
  give it a list of signals (`{ts, dir, vol}`), it returns trades. Entries at
  the real ask/bid, exits walked on the close-side feed, break-even, partial
  exits, spread filter, commission, swap with the triple-Wednesday rule, and
  equity-dependent sizing.
- **A worked example** (`examples/eurusd_wick_fade.py`): a complete EURUSD
  M15 strategy (cyclic RSI + rejection-wick fades + optional regime filter)
  showing how to go from bars to signals to tick-level results.

## Quickstart

```bash
git clone https://github.com/XScryer/tick-replay && cd tick-replay
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[ctrader]"                          # engine only: pip install -e .
```

### Try it without an account (synthetic data)

```bash
python examples/synthetic_ticks.py --days 90 --out data/SYNTH/raw
python -m tickreplay.data csv2bin --raw-dir data/SYNTH/raw --symbol SYNTH --bin-dir data/SYNTH/bin
python examples/eurusd_wick_fade.py --bin-dir data/SYNTH/bin \
    --from 2024-01-15 --to 2024-04-15 --preset wick_fade
```

Synthetic prices are a random walk — this only proves the pipeline runs; the
P&L means nothing.

### Real ticks from cTrader

1. **Create an Open API app** (free) at <https://openapi.ctrader.com/apps>
   and add a redirect URI such as `http://localhost:8899/callback`.
2. **Configure** — `cp .env.example .env` and fill in `OPEN_API_CLIENT_ID`,
   `OPEN_API_SECRET` and the same `CTRADER_REDIRECT_URI` you registered.
3. **Download** — the first run opens your browser to log in to cTrader;
   tokens are cached in `.env` (git-ignored) and refreshed automatically.

   ```bash
   python -m tickreplay.ctrader --symbol EURUSD \
       --from 2024-01-01 --to 2024-12-31 --out data/EURUSD/raw \
       --account-type demo            # or: live; --account <login> to pick one
   ```

   Interrupted? Run the same command again: finished days are skipped.
   `--probe` counts ticks without writing anything.

4. **Convert and check coverage**

   ```bash
   python -m tickreplay.data csv2bin --raw-dir data/EURUSD/raw --symbol EURUSD --bin-dir data/EURUSD/bin
   python -m tickreplay.data audit --bin-dir data/EURUSD/bin
   ```

5. **Run the example strategy** — pass *your* broker's costs:

   ```bash
   python examples/eurusd_wick_fade.py --bin-dir data/EURUSD/bin \
       --from 2024-02-01 --to 2024-12-31 \
       --commission 7.0 --swap-buy -7.0 --swap-sell 1.0
   ```

## Using the engine from your own code

```python
from tickreplay.data import load_both_sides
from tickreplay.replay import CostModel, ExitRules, ReplayEngine

ticks = load_both_sides("data/EURUSD/bin", start_ms=1704067200000, end_ms=1706745600000)
engine = ReplayEngine(*ticks["BID"], *ticks["ASK"],
                      costs=CostModel(commission_per_lot_rt=7.0, max_spread_pips=1.5))
signals = [{"ts": 1704204000000, "dir": "BUY", "vol": 0.10}]
trades = engine.run(signals, ExitRules(stop_pips=10, take_pips=20))
print(engine.trades_frame(trades))
```

## Engine semantics

- Signal `ts` = decision time (usually a bar close). The order executes on
  the first tick of either side at/after it, using the latest bid and ask at
  that instant.
- BUY fills at ask, SELL at bid. No latency, slippage or liquidity model.
- `max_spread_pips` skips signals when the spread is abnormally wide.
- Exits walk the close-side ticks (BUY watches bid, SELL watches ask). SL is
  checked before TP on each tick (conservative). A gapped stop fills at the
  worse tick price; TP fills at the TP level.
- Optional break-even move and partial close at `entry ± be_trigger_pips`.
- Costs: `pip_value_usd_per_lot × pips × lots` per exit leg, round-trip
  commission per lot, and swap per *swap day*: one per rollover, three for
  the Wednesday rollover, none for the weekend (`triple_swap_weekday`). The
  rollover is server midnight, by default the New York close
  (`ny_close_offset()`, UTC+2/+3 with US DST); pass `fixed_offset(h)`,
  `eu_broker_offset()` or your own function for other brokers.
- One position at a time. `vol` may be a number or a callable receiving
  `{"equity", "deposit", "trades"}`, so compounding sizing works.

## Data formats

**Raw CSV** (one per side, appended day by day): `time_ms,time,price`

```
1704067200000,2024.01.01 00:00:00.000,1.10432
```

**`.npz` per side per year** (`BID_2024.npz`, `ASK_2024.npz`): `ts` int64
epoch ms UTC, `px` float64. Bars are always built from ticks (bid side, like
MT4/MT5 charts), so the whole pipeline has a single source of truth.

## Limitations

- Tick history depth depends on the broker behind your account; always run
  `audit` before trusting a period.
- `pip_value_usd_per_lot` is constant (exact for XXX/USD pairs such as
  EURUSD; approximate for others).
- The example's default commission/swap are one broker's raw-spread values
  at one point in time; set yours with the CLI flags.
- `csv2bin` holds one side of one symbol in memory while merging; EURUSD is
  roughly 100–250k ticks per day per side, so convert very long histories in
  chunks. `load_ticks(start_ms=, end_ms=)` keeps backtests to the window you
  need.
- The example strategy is a demonstration of the workflow, not a claim of
  edge. Past ticks don't predict future ticks; this is not financial advice.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT — see [LICENSE](LICENSE).
