"""The spread as a hidden mean-reverting state: a state-space model for pairs trading.

Why this model
--------------
The original Kalman filter put the whole MA-V relationship into a random-walk
intercept and slope and treated everything else as white noise. Anything persistent
in the spread, which is exactly what a mean-reversion trader wants to trade, then gets
absorbed into the drifting intercept, and the "z-score" (one-day forecast error)
cannot become large.

Here the spread gets its own state with its own dynamics:

    y_t   = mu + beta_t * x_t + s_t + eps_t,     eps_t ~ N(0, R)          (observation)
    beta_t = beta_{t-1} + eta_t,                 eta_t ~ N(0, q_beta)     (slow hedge-ratio drift)
    s_t   = phi * s_{t-1} + zeta_t,              zeta_t ~ N(0, sigma2_s)  (mean-reverting spread)

y = log(MA), x = log(V). The Kalman filter now separates three things the original
model mixed together: slow structural change in the hedge ratio (beta_t), a
mean-reverting dislocation that should be traded (s_t), and daily noise (eps_t).
All five parameters are estimated by maximum likelihood on the training window only.

Signal: z_t = s_{t|t} / sd_train(s_{.|.}), the filtered spread divided by the standard
deviation of the filtered spread over the training window. (The filtered estimate is
shrunk toward zero relative to the true state, so dividing by the model's stationary
sd, sqrt(sigma2_s / (1 - phi^2)), would understate how unusual a dislocation is.)
The filtered s_{t|t} uses data through day t; positions are taken at the close of t
and earn day t+1's return.

Identification. A drifting hedge ratio and a slowly mean-reverting spread are close to
observationally equivalent over short horizons. Left free, maximum likelihood (which
scores one-day-ahead forecasts) lets beta_t absorb the persistent part and sets phi
near 0.3, leaving nothing to trade. q_beta is therefore treated as a modeling choice:
it is profiled over a grid, and the value used out of sample is the one with the best
TRAINING-period strategy Sharpe. Test-period results for every other value are reported
as sensitivity, not as the result.

The half-life of a dislocation is ln(2) / -ln(phi) trading days.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel

PHI_CAP = 0.9995
BURN_IN = 63  # days of training filter output ignored while the filter settles


class LatentSpread(MLEModel):
    param_names_ = ["mu", "q_beta", "phi", "sigma2_s", "R"]

    def __init__(self, y: np.ndarray, x: np.ndarray, beta0: float, mu0: float):
        super().__init__(np.asarray(y, float), k_states=2, k_posdef=2)
        self._x = np.asarray(x, float)
        self._beta0, self._mu0 = beta0, mu0
        design = np.zeros((1, 2, len(y)))
        design[0, 0, :] = self._x
        design[0, 1, :] = 1.0
        self["design"] = design
        self["selection"] = np.eye(2)
        self["transition"] = np.eye(2)

    @property
    def param_names(self):
        return self.param_names_

    @property
    def start_params(self):
        return np.array([self._mu0, 1e-6, 0.97, 1e-4, 1e-5])

    def transform_params(self, u):
        p = np.array(u, float)
        p[1], p[3], p[4] = np.exp(u[1]), np.exp(u[3]), np.exp(u[4])
        p[2] = PHI_CAP * np.tanh(u[2])
        return p

    def untransform_params(self, p):
        u = np.array(p, float)
        u[1], u[3], u[4] = np.log(p[1]), np.log(p[3]), np.log(p[4])
        u[2] = np.arctanh(p[2] / PHI_CAP)
        return u

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        mu, qb, phi, s2, R = params
        self["obs_intercept"] = np.array([[mu]])
        self["obs_cov"] = np.array([[R]])
        self["transition"] = np.array([[1.0, 0.0], [0.0, phi]])
        self["state_cov"] = np.diag([qb, s2])
        # beta starts at the training OLS slope with a loose prior; s starts stationary
        self.ssm.initialize_known(np.array([self._beta0, 0.0]),
                                  np.diag([0.05**2, s2 / (1 - phi**2)]))


def fit(y_train, x_train, beta0, mu0):
    mod = LatentSpread(y_train, x_train, beta0, mu0)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for phi0 in (0.95, 0.99):
            sp = mod.start_params.copy()
            sp[2] = phi0
            r = mod.fit(start_params=sp, disp=False, maxiter=2000, method="lbfgs")
            r2 = mod.fit(start_params=r.params, disp=False, maxiter=4000, method="nm")
            r = r2 if r2.llf > r.llf else r
            if best is None or r.llf > best.llf:
                best = r
    return best


def describe(res) -> dict:
    mu, qb, phi, s2, R = res.params
    return {"mu": mu, "q_beta": qb, "phi": phi, "sigma2_s": s2, "R": R,
            "half_life_days": float(np.log(2) / -np.log(phi)),
            "spread_sd": float(np.sqrt(s2 / (1 - phi**2))), "loglik": float(res.llf)}


def filter_with(y_all, x_all, beta0, mu0, params):
    mod = LatentSpread(y_all, x_all, beta0, mu0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.filter(params)


def signal(train: pd.DataFrame, test: pd.DataFrame, logp: pd.DataFrame, ya="MA", xa="V", res=None):
    """Fit on train; filter through train+test with frozen parameters; return (beta, z, info)."""
    import statsmodels.api as sm

    ols = sm.OLS(train[ya], sm.add_constant(train[xa])).fit()
    a, b = ols.params.iloc[0], ols.params.iloc[1]
    if res is None:
        res = fit(train[ya].values, train[xa].values, b, a)
    full = logp.loc[: test.index[-1]]
    f = filter_with(full[ya].values, full[xa].values, b, a, res.params)
    info = describe(res)
    beta = pd.Series(f.filtered_state[0], index=full.index)
    s = pd.Series(f.filtered_state[1], index=full.index)
    sd_train = s.loc[train.index[BURN_IN:]].std()
    info["filtered_spread_sd_train"] = float(sd_train)
    z = s / sd_train
    return beta.loc[test.index], z.loc[test.index], info, s, beta


def fit_fixed_qbeta(y_train, x_train, beta0, mu0, q_beta: float):
    """Profile likelihood: hold the hedge-ratio drift q_beta fixed, estimate the rest."""
    mod = LatentSpread(y_train, x_train, beta0, mu0)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mod.fix_params({"q_beta": q_beta}):
            for phi0 in (0.9, 0.99):
                sp = np.array([mu0, 0.97, 1e-4, 1e-5])
                sp[1] = phi0
                r = mod.fit(start_params=sp, disp=False, maxiter=2000, method="lbfgs")
                r2 = mod.fit(start_params=r.params[mod._free_params_index], disp=False, maxiter=4000, method="nm")
                r = r2 if r2.llf > r.llf else r
                if best is None or r.llf > best.llf:
                    best = r
    return best
