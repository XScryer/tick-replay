# Changelog

## 0.2.0

### Fixed
- Swap model: rollovers now follow the spot-FX convention (triple swap on
  Wednesday, nothing for the weekend). Previously Friday→Monday was charged
  3 days and Wednesday→Thursday 1 day.
- Default server clock is now the New York close (UTC+2/+3 following US
  daylight saving); `eu_broker_offset()` remains available.
- Entry quote includes the tick that arrives at the decision instant
  (previously the quote just before it was used).
- Downloader connects to the demo or live host matching the chosen account
  (`--account-type`, `--account`); it used to always connect to live.
- Symbol `digits` is read from `ProtoOASymbolByIdReq` (the light symbol list
  has no such field) and prices use the fixed 1/100000 API scale.
- `Quickstart` works from a clean checkout (`pip install -e ".[ctrader]"`).
- `audit` only counts missing weekdays inside the covered span.

### Added
- OAuth redirect URI configurable via `CTRADER_REDIRECT_URI`; automatic token
  refresh (on expiry and when the server rejects the token); `--reauth`.
- `load_ticks(..., start_ms=, end_ms=)` loads only a time window.
- `Trade.swap_days`, `swap_days()`, `rollover_times()`, `ny_close_offset()`.
- Tests for the downloader helpers, data layer, regime classifier, swap model
  and more engine edge cases; GitHub Actions CI (ruff + pytest + smoke test).

### Changed
- Example presets renamed to `wick_fade_regime` / `wick_fade`.
- `requirements.txt` removed; dependencies live in `pyproject.toml`.

## 0.1.0
- Initial release.
