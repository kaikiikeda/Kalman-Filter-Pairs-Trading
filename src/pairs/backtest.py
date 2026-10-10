"""Trading rule, P&L accounting and performance metrics (same rules as notebook 04).

Rule: enter short the spread when z > +entry, long when z < -entry; exit when z
crosses back through 0. Position at the close of day t earns the return of day t+1.
Weights: MA leg = pos / (1 + |beta|), V leg = -pos * beta / (1 + |beta|), so gross
exposure is 1. Costs: `cost_bps` per unit of turnover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENTRY = 2.0
COST_BPS = 5.0


def positions(z: pd.Series, entry: float = ENTRY, exit_level: float = 0.0) -> pd.Series:
    pos, out = 0, []
    for v in z.to_numpy():
        if np.isnan(v):
            out.append(pos)
            continue
        if pos == 0:
            if v > entry:
                pos = -1
            elif v < -entry:
                pos = 1
        elif pos == 1 and v >= exit_level:
            pos = 0
        elif pos == -1 and v <= -exit_level:
            pos = 0
        out.append(pos)
    return pd.Series(out, index=z.index, dtype=float)


def weights(pos: pd.Series, beta: pd.Series) -> pd.DataFrame:
    d = 1 + beta.abs()
    return pd.DataFrame({"MA": pos / d, "V": -pos * beta / d}, index=pos.index)


def run(z: pd.Series, beta: pd.Series, rets: pd.DataFrame, entry: float = ENTRY,
        cost_bps: float = COST_BPS) -> dict:
    pos = positions(z, entry)
    w = weights(pos, beta)
    prev = w.shift(1).fillna(0)
    gross = prev["MA"] * rets["MA"] + prev["V"] * rets["V"]
    turnover = (w - w.shift(1).fillna(0)).abs().sum(axis=1)
    net = gross - cost_bps / 1e4 * turnover.shift(1).fillna(0)
    return {"ret": net.fillna(0.0), "gross": gross.fillna(0.0), "pos": pos, "turnover": turnover, "w": w}


def trades(pos: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """(entry date, exit date, direction) for every round trip."""
    out, start, d = [], None, 0
    for t, p in pos.items():
        if d == 0 and p != 0:
            start, d = t, int(p)
        elif d != 0 and p != d:
            out.append((start, t, d))
            start, d = (t, int(p)) if p != 0 else (None, 0)
    if d != 0:
        out.append((start, pos.index[-1], d))
    return out


def metrics(res: dict) -> dict:
    r = res["ret"]
    eq = (1 + r).cumprod()
    n = len(r)
    sd = r.std()
    tr = trades(res["pos"])
    trade_rets = [float((1 + r.loc[a:b].iloc[1:]).prod() - 1) for a, b, _ in tr] if tr else []
    return {
        "Annual return": eq.iloc[-1] ** (252 / n) - 1,
        "Annual vol": sd * np.sqrt(252),
        "Sharpe": r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "Max drawdown": (eq / eq.cummax() - 1).min(),
        "Trades": len(tr),
        "Days in market": float((res["pos"] != 0).mean()),
        "Hit rate": float(np.mean([x > 0 for x in trade_rets])) if trade_rets else np.nan,
        "Avg holding (days)": float(np.mean([res["pos"].index.get_loc(b) - res["pos"].index.get_loc(a)
                                             for a, b, _ in tr])) if tr else np.nan,
        "Turnover": res["turnover"].sum(),
    }


def sharpe_ci(r: pd.Series, n_boot: int = 5000, block: int = 21, seed: int = 0) -> tuple[float, float, float]:
    """Stationary-block bootstrap 95% CI for the annualized Sharpe ratio, and P(Sharpe <= 0)."""
    rng = np.random.default_rng(seed)
    x = r.to_numpy()
    n = len(x)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            L = rng.geometric(1 / block)
            idx.extend((s + np.arange(L)) % n)
        b = x[np.array(idx[:n])]
        out[i] = b.mean() / b.std() * np.sqrt(252) if b.std() > 0 else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())
