"""Market-regime classifier — the strategy's "opinion" per closed bar.

For every bar the classifier says which state the market is in:

    opinion[i] ==  2  confirmed UP-leg    -> blocks fade entries
    opinion[i] ==  3  confirmed DOWN-leg  -> blocks fade entries
    opinion[i] ==  1  confirmed RANGE     -> dead/choppy market
    opinion[i] == -1  NO OPINION (transition/ambiguous) -> default: trade

CAUSALITY GUARANTEE: opinion[i] uses ONLY bars with index <= i (all already
closed). It never reads a future bar. The `pers` consensus looks BACKWARD
(bars i-pers+1..i), never forward.

HOW TO USE:

    import numpy as np
    from tickreplay.regime import regime_opinion

    # bars: dict of numpy arrays with "t0" (UTC ms) and "o","h","l","c" (price)
    opinion = regime_opinion(bars)                 # default parameters
    # opinion is an int8 array of the same length as bars["c"]

Parameters (all optional; defaults chosen for EURUSD M15):

    K        analysis window in bars           (12 = 3h on M15)
    net_min  minimum net move in pips          (10.0)
    er_hi    minimum efficiency for a leg      (0.45)
    er_lo    maximum efficiency for a range    (0.25)
    rng_max  maximum range in pips for a range (25.0)
    pers     consecutive bars with same label  (6)
    pip      instrument pip size               (0.0001 EURUSD; index CFDs differ)

Definitions (all in pips, computed on closes):
    net[i]  = close[i] - close[i-K]          (net move over K bars)
    path[i] = sum |close[j]-close[j-1]|     (total path, j=i-K+1..i)
    er[i]   = |net[i]| / path[i]             (0=chop, 1=straight line)
    rng[i]  = max(high)-min(low) over K bars

Raw per-bar label (needs i >= K):
    PER_UP  if net >= +net_min  and er >= er_hi
    PER_DN  if net <= -net_min  and er >= er_hi
    RANGE   if er <= er_lo      and rng <= rng_max
    (otherwise: no opinion)

Final opinion exists only if the last `pers` bars ALL carry the same label
>= 0 — a backward consensus. Otherwise -1.

Honest limitations:
    - Measures recent ACTIVITY/direction, does not predict the future.
    - A leg is only confirmed after it has already moved (~1-3h lag inherent).
    - Reduces damage for counter-trend entries; does not create edge in
      structureless markets.
    - Defaults were selected on EURUSD M15 — for another instrument or
      timeframe, re-validate (start by scaling net_min/rng_max to the
      typical range of that timeframe).
"""
import numpy as np
import pandas as pd


def regime_opinion(b, K=12, net_min=10.0, er_hi=0.45, er_lo=0.25,
                   rng_max=25.0, pers=6, pip=0.0001):
    """Return an int8 array with the regime opinion per bar (causal).

    b: dict with "o","h","l","c" (float price arrays) — "t0" is not used.
    """
    pc = np.asarray(b["c"], dtype=np.float64) / pip
    ph = np.asarray(b["h"], dtype=np.float64) / pip
    pl = np.asarray(b["l"], dtype=np.float64) / pip
    n = len(pc)

    net = np.full(n, np.nan)
    net[K:] = pc[K:] - pc[:-K]
    d1 = np.abs(np.diff(pc, prepend=pc[0]))
    pre = np.concatenate([[0.0], np.cumsum(d1)])
    path = np.full(n, np.nan)
    path[K:] = pre[K + 1:] - pre[1:n - K + 1]
    er = np.abs(net) / np.where(path == 0, np.nan, path)
    rng = (pd.Series(ph).rolling(K).max()
           - pd.Series(pl).rolling(K).min()).to_numpy()

    raw = np.full(n, -1, dtype=np.int8)
    raw[(er <= er_lo) & (rng <= rng_max)] = 1
    raw[(net >= net_min) & (er >= er_hi)] = 2
    raw[(net <= -net_min) & (er >= er_hi)] = 3

    # backward consensus: `pers` consecutive bars with the same label
    opinion = np.full(n, -1, dtype=np.int8)
    for v in (1, 2, 3):
        m = (raw == v).astype(np.int32)
        cs = np.concatenate([[0], np.cumsum(m)])
        ok = (cs[pers:] - cs[:-pers]) == pers
        opinion[np.where(ok)[0] + pers - 1] = v
    return opinion


# human-readable state names
LABELS = {-1: "NO_OPINION", 1: "RANGE", 2: "UP_LEG", 3: "DOWN_LEG"}
