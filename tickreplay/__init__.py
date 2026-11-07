"""tick-replay: real tick data + tick-level backtesting.

Pipeline:
    1. tickreplay.ctrader  -> download historical bid/ask ticks (cTrader Open API)
    2. tickreplay.data     -> convert CSVs to per-year .npz, build OHLC bars
    3. tickreplay.replay   -> simulate trades tick-by-tick on real quotes
    4. tickreplay.regime   -> optional causal market-regime classifier
"""

__version__ = "0.2.0"
