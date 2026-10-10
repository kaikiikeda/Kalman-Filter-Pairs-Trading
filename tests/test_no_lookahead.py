"""Causality tests: corrupting prices after a cutoff must not change any signal before it.

Also checks that the refactored code reproduces the original notebook's numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pairs import backtest as B  # noqa: E402
from pairs import hedge as H  # noqa: E402
from pairs import latent_spread as LS  # noqa: E402
from pairs.data import DATA, split  # noqa: E402


def synthetic(n=1400, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.015, n)) + 4.0
    beta = 1.1 + np.cumsum(rng.normal(0, 3e-4, n))
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.97 * s[t - 1] + rng.normal(0, 0.006)
    y = 0.2 + beta * x + s + rng.normal(0, 0.002, n)
    idx = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame({"V": x, "MA": y}, index=idx)


@pytest.fixture(scope="module")
def setup():
    lp = synthetic()
    tr, te = lp.iloc[:900], lp.iloc[900:]
    cut = te.index[200]
    bad = lp.copy()
    bad.loc[bad.index > cut, :] += np.random.default_rng(1).normal(0, 0.3, (int((bad.index > cut).sum()), 2))
    return lp, tr, te, bad, cut


def signals(lp, tr, te, mle, lat_res):
    te_ = lp.loc[te.index]
    return {
        "static": H.static_ols(tr, te_),
        "rolling": H.rolling_ols(lp, te.index),
        "kalman_original": H.kalman_original(tr, te_),
        "kalman_mle_innovation": H.kalman_mle_innovation(tr, te_, params=mle),
        "kalman_beta": H.kalman_beta(tr, te_, lp, params=mle),
        "latent": H.Signal("latent", *LS.signal(tr, te_, lp, res=lat_res)[:2]),
    }


def test_signals_ignore_the_future(setup):
    lp, tr, te, bad, cut = setup
    mle = H.fit_kalman_mle(tr["MA"].values, tr["V"].values, np.array([0.2, 1.1]))
    a, b, _ = H._ols(tr["MA"], tr["V"])
    lat = LS.fit_fixed_qbeta(tr["MA"].values, tr["V"].values, b, a, 1e-7)
    good, worse = signals(lp, tr, te, mle, lat), signals(bad, tr, te, mle, lat)
    for k in good:
        for attr in ("z", "beta"):
            g, w = getattr(good[k], attr), getattr(worse[k], attr)
            np.testing.assert_allclose(g.loc[:cut], w.loc[:cut], atol=1e-10, err_msg=f"{k}.{attr}")
            if (k, attr) == ("static", "beta"):
                continue  # constant by construction
            assert not np.allclose(g.loc[cut:].iloc[2:], w.loc[cut:].iloc[2:]), f"{k}.{attr}: corruption had no effect"


def test_positions_trade_next_day():
    z = pd.Series([0, 3, 3, 0, 0], index=pd.bdate_range("2020-01-01", periods=5), dtype=float)
    beta = pd.Series(1.0, index=z.index)
    rets = pd.DataFrame({"MA": [0, 0, 0.10, 0, 0], "V": [0, 0, 0, 0, 0]}, index=z.index)
    res = B.run(z, beta, rets, cost_bps=0)
    # short entered at close of day 1, so day 2's +10% MA move is a loss of 10% * 0.5 weight
    assert res["gross"].iloc[1] == 0
    assert res["gross"].iloc[2] == pytest.approx(-0.05)


@pytest.mark.skipif(not DATA.exists(), reason="needs data/prices.csv")
def test_reproduces_original_notebook():
    from pairs.data import load_prices

    p = load_prices()[["V", "MA"]]
    lp = np.log(p)
    tr, te = split(lp)
    rets = p.pct_change().loc[te.index]
    st = B.metrics(B.run(*_zb(H.static_ols(tr, te)), rets))
    ro = B.metrics(B.run(*_zb(H.rolling_ols(lp, te.index)), rets))
    ko = B.metrics(B.run(*_zb(H.kalman_original(tr, te)), rets))
    assert st["Annual return"] == pytest.approx(0.0145, abs=5e-4) and st["Trades"] == 4
    assert ro["Sharpe"] == pytest.approx(0.6696, abs=5e-3) and ro["Trades"] == 13
    assert ko["Trades"] == 0


def _zb(s):
    return s.z, s.beta
