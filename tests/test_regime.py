import numpy as np

from tickreplay.regime import regime_opinion

PIP = 1e-4


def _bars(closes):
    c = np.asarray(closes, dtype=float)
    return {"o": c, "h": c + 0.5 * PIP, "l": c - 0.5 * PIP, "c": c}


def test_trend_up_down_and_range_labels():
    up = 1.1 + np.arange(60) * 2 * PIP                 # straight 2-pip steps
    down = 1.1 - np.arange(60) * 2 * PIP
    chop = 1.1 + np.tile([0, PIP], 30)                 # 1-pip zig-zag
    assert regime_opinion(_bars(up))[-1] == 2
    assert regime_opinion(_bars(down))[-1] == 3
    assert regime_opinion(_bars(chop))[-1] == 1
    # no opinion before the window and the consensus are complete
    assert (regime_opinion(_bars(up))[:12 + 6 - 1] == -1).all()


def test_opinion_is_causal():
    rng = np.random.default_rng(1)
    closes = 1.1 + np.cumsum(rng.normal(0, 2 * PIP, 400))
    base = regime_opinion(_bars(closes))
    changed = closes.copy()
    changed[250:] += np.cumsum(rng.normal(0, 10 * PIP, 150))   # rewrite the future
    after = regime_opinion(_bars(changed))
    assert (base[:250] == after[:250]).all()
