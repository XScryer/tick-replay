# tick-replay

**Real tick data + tick-level backtesting — free, via the cTrader Open API.**

Most free backtesting runs on *simulated* ticks: the platform invents prices
between bars and your fills are fiction. This repo gives you the real thing:
historical **bid and ask tick data** (the actual quotes your broker streamed)
plus an execution engine that replays trades tick-by-tick with real spread,
commission and swap.

```
cTrader Open API ──► raw CSVs ──► per-year .npz ──► bars / signals ──► tick replay ──► trades
   (free, demo OK)    (resumable)   (fast load)      (your strategy)    (real fills)   (report)
```

## What problem does it solve?

- **"I want to backtest on real ticks but tick data is expensive."**
  cTrader's Open API serves years of real bid/ask history on a free demo
  account. This downloader handles the pagination quirks (delta-encoded
  timestamps, newest-first chunks, weekly request limits) and resumes
  interrupted multi-year downloads automatically.
- **"My bar-based backtest keeps lying to me."**
  `tickreplay.replay` executes signals on actual tick sequences: entry at the
  real ask/bid, exits evaluated tick-by-tick on the close-side feed,
  break-even triggers, partial exits, spread filter, commission and
  broker-day swap rollover — the same conventions a broker's matching engine
  uses.
- **"I don't want to write a trading robot to test an idea."**
  The engine is strategy-agnostic: you produce a list of signals
  (`{ts, dir, vol}`), it returns trades. `examples/eurusd_wick_fade.py` is a
  complete worked strategy you can copy and modify.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your cTrader Open API app credentials
```

1. **Get API credentials** (free): create an app at
   <https://openapi.ctrader.com/apps>, then authorize a **demo** cTrader
   account when the downloader first opens your browser.

2. **Download ticks** (writes resumable CSVs):

   ```bash
   python -m tickreplay.ctrader --symbol EURUSD \
       --from 2020-01-01 --to 2024-12-31 --out data/EURUSD/raw
   ```

3. **Convert to compact .npz** (per year, per side):

   ```bash
   python -m tickreplay.data csv2bin \
       --raw-dir data/EURUSD/raw --symbol EURUSD --bin-dir data/EURUSD/bin
   ```

4. **Run the example strategy** (EURUSD M15 wick-fade, real ticks):

   ```bash
   python examples/eurusd_wick_fade.py \
       --bin-dir data/EURUSD/bin --de 2020-01-01 --ate 2024-12-31
   ```

5. **No account yet? Try synthetic data** to smoke-test the engine:

   ```bash
   python examples/synthetic_ticks.py --days 10 --out data/SYNTH/raw
   python -m tickreplay.data csv2bin --raw-dir data/SYNTH/raw --symbol SYNTH --bin-dir data/SYNTH/bin
   python examples/eurusd_wick_fade.py --bin-dir data/SYNTH/bin --de 2024-01-02 --ate 2024-01-09
   ```

## Repository layout

```
tickreplay/
  ctrader.py    cTrader Open API tick downloader (OAuth, pagination, resume)
  data.py       CSV→npz conversion, tick→bar aggregation, coverage audit
  replay.py     tick-level execution engine (fills, exits, costs, sizing)
  regime.py     causal market-regime classifier (optional signal filter)
examples/
  synthetic_ticks.py    fake tick generator for smoke tests
  eurusd_wick_fade.py   complete worked strategy (cRSI + wick fades + regime gate)
```

## Data formats

**Raw CSV** (one per side, appended day by day):

```
time_ms,time,price
1704067200000,2024.01.01 00:00:00.000,1.10432
```

**`.npz` per year per side** (`BID_2024.npz`, `ASK_2024.npz`):

- `ts`: int64 epoch milliseconds UTC
- `px`: float64 price

Bars are always built from ticks (`python -m tickreplay.data bars`), so the
whole pipeline has a single source of truth.

## Engine semantics

- Signal `ts` = decision time (usually a bar close). The engine takes the
  first quote on each side at/after that instant.
- BUY fills at ask, SELL at bid — real spread at that moment.
- `max_spread_pips` skips signals when the spread is abnormally wide.
- Exits walk the close-side ticks (BUY watches bid, SELL watches ask);
  SL is checked before TP on each tick (conservative intra-tick rule).
- Optional break-even move and partial close when price reaches
  `entry ± be_trigger_pips`.
- Costs: `pip_value_usd_per_lot` (quote-currency dependent — EURUSD ≈ 10),
  round-trip commission per lot, swap per broker-day boundary crossed
  (server midnight; EU-time brokers covered by default, override
  `rollover_offset` for others).
- One position at a time. `vol` may be a number or a callable receiving the
  running equity — compounding sizing works.

## Honest limitations

- Tick history depth depends on the broker/account behind your API token —
  check coverage with `python -m tickreplay.data audit --bin-dir data/EURUSD/bin`.
- Swap/commission defaults in the example are *measured* values for one
  broker (Pepperstone Razor EURUSD); always set yours via CLI flags.
- Bars built from ticks use the **bid** convention (same as MT5).
- Large histories need RAM: EURUSD is ~100–250k ticks/day/side. Load only
  the years you need (`--years` on the loader).
- This is a research tool, not financial advice. Past ticks don't predict
  future ticks.

## Requirements

Python 3.10+, `numpy`, `pandas`, and — only for the downloader —
`ctrader-open-api` + `twisted` (see `requirements.txt`).

## License

MIT — see [LICENSE](LICENSE).
