"""Hedge-ratio estimators and the spread signals they produce.

Notation: y = log(MA), x = log(V). Every estimator returns, for each test day t,
a hedge ratio beta_t and a standardized signal z_t computed ONLY from data up to
and including day t. Positions taken at the close of t earn the return of t+1.

Estimators
----------
static_ols        alpha, beta fitted once on the training window; z from training mean/std
rolling_ols       alpha, beta refit every day on the previous 126 days
kalman_original   the original notebook's Kalman filter (random-walk alpha and beta,
                  hand-set Q = 1e-5 I and R = training OLS residual variance),
                  z = one-step innovation / sqrt(innovation variance)
kalman_beta       Kalman filter for the hedge ratio only (alpha, beta random walks with
                  Q, R estimated by maximum likelihood on the training window), then the
                  spread y - beta_t x is z-scored against its own trailing 63-day mean/std.
                  This is the common practitioner construction.
kalman_latent     state-space model with the spread itself as a hidden mean-reverting
                  state (see latent_spread.py). z = filtered spread / its stationary std.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROLL_WINDOW = 126
Z_WINDOW = 63


@dataclass
class Signal:
    name: str
    beta: pd.Series  # hedge ratio used to size the V leg
    z: pd.Series  # standardized spread; trade when |z| is large
    extra: dict | None = None


def _ols(y: pd.Series, x: pd.Series):
    m = sm.OLS(y, sm.add_constant(x)).fit()
    return m.params.iloc[0], m.params.iloc[1], m


# --------------------------------------------------------------------- OLS
def static_ols(train: pd.DataFrame, test: pd.DataFrame, ya="MA", xa="V") -> Signal:
    a, b, _ = _ols(train[ya], train[xa])
    tr_spread = train[ya] - a - b * train[xa]
    z = (test[ya] - a - b * test[xa] - tr_spread.mean()) / tr_spread.std()
    return Signal("Static OLS", pd.Series(b, index=test.index), z, {"alpha": a, "beta": b})


def rolling_ols(logp: pd.DataFrame, test_index: pd.DatetimeIndex, ya="MA", xa="V",
                window: int = ROLL_WINDOW) -> Signal:
    betas, zs = [], []
    for date in test_index:
        i = logp.index.get_loc(date)
        hist = logp.iloc[i - window:i]  # the previous `window` days, excluding today
        a, b, m = _ols(hist[ya], hist[xa])
        resid_t = logp[ya].iloc[i] - a - b * logp[xa].iloc[i]
        zs.append((resid_t - m.resid.mean()) / m.resid.std())
        betas.append(b)
    return Signal("Rolling OLS", pd.Series(betas, index=test_index), pd.Series(zs, index=test_index))


# ------------------------------------------------------------------ Kalman
def kalman_regression(y: np.ndarray, x: np.ndarray, theta0: np.ndarray, P0: np.ndarray,
                      Q: np.ndarray, R: float):
    """Time-varying regression y_t = a_t + b_t x_t + v_t with random-walk [a_t, b_t].

    Returns one-step-ahead (predicted) states, innovations e_t and their variances S_t.
    The prediction for day t uses only data through t-1, so e_t / sqrt(S_t) is causal.
    Uses the Joseph-form covariance update, as in the original notebook.
    """
    n = len(y)
    theta, P, I = theta0.astype(float).copy(), P0.astype(float).copy(), np.eye(2)
    th_pred = np.empty((n, 2))
    e = np.empty(n)
    S = np.empty(n)
    ll = 0.0
    for t in range(n):
        H = np.array([1.0, x[t]])
        P_pred = P + Q
        th_pred[t] = theta
        e[t] = y[t] - H @ theta
        S[t] = H @ P_pred @ H + R
        K = P_pred @ H / S[t]
        theta = theta + K * e[t]
        A = I - np.outer(K, H)
        P = A @ P_pred @ A.T + np.outer(K, K) * R
        ll += -0.5 * (np.log(2 * np.pi * S[t]) + e[t] ** 2 / S[t])
    return th_pred, e, S, theta, P, ll


def kalman_original(train: pd.DataFrame, test: pd.DataFrame, ya="MA", xa="V",
                    q: float = 1e-5) -> Signal:
    """Exact replica of notebook 04: burn in on train, then trade the test innovations."""
    a, b, m = _ols(train[ya], train[xa])
    R = m.mse_resid
    Q = q * np.eye(2)
    _, _, _, th, P, _ = kalman_regression(train[ya].values, train[xa].values,
                                          np.array([a, b]), np.eye(2), Q, R)
    th_pred, e, S, *_ = kalman_regression(test[ya].values, test[xa].values, th, P, Q, R)
    z = pd.Series(e / np.sqrt(S), index=test.index)
    return Signal("Kalman (original spec)", pd.Series(th_pred[:, 1], index=test.index), z,
                  {"R": R, "q": q, "innovation_sd": float(np.std(e)), "sqrt_S": float(np.sqrt(S).mean())})


def fit_kalman_mle(y: np.ndarray, x: np.ndarray, theta0: np.ndarray):
    """Estimate Q = diag(qa, qb) and R by maximum likelihood (log-parameterized)."""
    from scipy.optimize import minimize

    def nll(p):
        qa, qb, R = np.exp(p)
        *_, ll = kalman_regression(y, x, theta0, np.eye(2), np.diag([qa, qb]), R)
        return -ll

    best = None
    for start in ([-12, -12, -9], [-9, -11, -8], [-14, -10, -7]):
        r = minimize(nll, start, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    qa, qb, R = np.exp(best.x)
    return {"q_alpha": qa, "q_beta": qb, "R": R, "loglik": -best.fun}


def kalman_mle_innovation(train: pd.DataFrame, test: pd.DataFrame, ya="MA", xa="V",
                          params: dict | None = None) -> Signal:
    """Original construction (trade the one-step innovation) but with Q and R estimated by MLE."""
    a, b, _ = _ols(train[ya], train[xa])
    if params is None:
        params = fit_kalman_mle(train[ya].values, train[xa].values, np.array([a, b]))
    Q, R = np.diag([params["q_alpha"], params["q_beta"]]), params["R"]
    _, _, _, th, P, _ = kalman_regression(train[ya].values, train[xa].values, np.array([a, b]), np.eye(2), Q, R)
    th_pred, e, S, *_ = kalman_regression(test[ya].values, test[xa].values, th, P, Q, R)
    return Signal("Kalman (MLE) innovation", pd.Series(th_pred[:, 1], index=test.index),
                  pd.Series(e / np.sqrt(S), index=test.index), params)


def kalman_beta(train: pd.DataFrame, test: pd.DataFrame, logp: pd.DataFrame, ya="MA", xa="V",
                z_window: int = Z_WINDOW, params: dict | None = None) -> Signal:
    """Kalman hedge ratio (MLE noise), spread z-scored on a trailing window."""
    a, b, _ = _ols(train[ya], train[xa])
    if params is None:
        params = fit_kalman_mle(train[ya].values, train[xa].values, np.array([a, b]))
    Q, R = np.diag([params["q_alpha"], params["q_beta"]]), params["R"]
    full = logp.loc[: test.index[-1]]
    th_pred, *_ = kalman_regression(full[ya].values, full[xa].values, np.array([a, b]), np.eye(2), Q, R)
    beta = pd.Series(th_pred[:, 1], index=full.index)  # beta known BEFORE seeing day t
    spread = full[ya] - beta * full[xa]
    mu = spread.shift(1).rolling(z_window).mean()  # trailing stats exclude today
    sd = spread.shift(1).rolling(z_window).std()
    z = ((spread - mu) / sd).loc[test.index]
    return Signal("Kalman beta + rolling z", beta.loc[test.index], z, params)
