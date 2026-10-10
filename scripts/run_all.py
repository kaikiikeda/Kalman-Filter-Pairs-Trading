"""Run the full study and write every number and figure used in the README and report.

Usage:  python scripts/run_all.py         (~3 minutes)
Outputs: results/tables/*.csv, results/figures/*.png, results/summary.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from pairs import backtest as B  # noqa: E402
from pairs import hedge as H  # noqa: E402
from pairs import latent_spread as LS  # noqa: E402
from pairs import plots as P  # noqa: E402
from pairs.data import CANDIDATE_PAIRS, load_prices, split  # noqa: E402

TAB, FIG = ROOT / "results" / "tables", ROOT / "results" / "figures"
QB_GRID = [0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 2e-6]
ORDER = ["Static OLS", "Rolling OLS", "Kalman (original spec)", "Kalman (MLE) innovation",
         "Kalman beta + rolling z", "Latent-spread state space"]


def latent_profile(tr, te, lp, rets, ya, xa):
    """Profile q_beta; choose it by TRAINING Sharpe; return chosen signal + the grid table."""
    ols = sm.OLS(tr[ya], sm.add_constant(tr[xa])).fit()
    a, b = ols.params.iloc[0], ols.params.iloc[1]
    rows, cache = [], {}
    for qb in QB_GRID:
        res = LS.fit_fixed_qbeta(tr[ya].values, tr[xa].values, b, a, qb)
        beta_te, z_te, info, s_full, beta_full = LS.signal(tr, te, lp, ya, xa, res=res)
        z_full = s_full / info["filtered_spread_sd_train"]
        tri = tr.index[LS.BURN_IN:]
        m_tr = B.metrics(B.run(z_full.loc[tri], beta_full.loc[tri], rets.loc[tri]))
        m_te = B.metrics(B.run(z_te, beta_te, rets.loc[te.index]))
        rows.append({"q_beta": qb, "loglik": info["loglik"], "phi": info["phi"],
                     "half_life_days": info["half_life_days"], "R": info["R"],
                     "train_sharpe": m_tr["Sharpe"], "train_trades": m_tr["Trades"],
                     "test_sharpe": m_te["Sharpe"], "test_return": m_te["Annual return"],
                     "test_maxdd": m_te["Max drawdown"], "test_trades": m_te["Trades"]})
        cache[qb] = (beta_te, z_te, info, s_full, beta_full)
    # free MLE for reference (all parameters estimated)
    free = LS.fit(tr[ya].values, tr[xa].values, b, a)
    fd = LS.describe(free)
    grid = pd.DataFrame(rows)
    pick = grid.loc[grid["train_sharpe"].fillna(-np.inf).idxmax(), "q_beta"]
    beta_te, z_te, info, s_full, beta_full = cache[pick]
    sig = H.Signal("Latent-spread state space", beta_te, z_te, {**info, "q_beta": pick})
    return sig, grid, fd, s_full, beta_full


def run_pair(prices, ya, xa, verbose=True):
    p = prices[[xa, ya]]
    lp = np.log(p)
    tr, te = split(lp)
    # backtest.run expects the dependent leg as "MA" and the hedge leg as "V"
    rets = p.pct_change().rename(columns={ya: "MA", xa: "V"})
    sigs = [H.static_ols(tr, te, ya, xa), H.rolling_ols(lp, te.index, ya, xa),
            H.kalman_original(tr, te, ya, xa)]
    a, b, _ = H._ols(tr[ya], tr[xa])
    mle = H.fit_kalman_mle(tr[ya].values, tr[xa].values, np.array([a, b]))
    sigs += [H.kalman_mle_innovation(tr, te, ya, xa, mle), H.kalman_beta(tr, te, lp, ya, xa, params=mle)]
    lat, grid, free, s_full, beta_full = latent_profile(tr, te, lp, rets, ya, xa)
    sigs.append(lat)
    r_te = rets.loc[te.index]
    results = {s.name: B.run(s.z, s.beta, r_te) for s in sigs}
    if verbose:
        print(f"  {ya}/{xa}: done")
    return dict(sigs=sigs, results=results, grid=grid, free=free, mle=mle, tr=tr, te=te, lp=lp,
                rets=r_te, s_full=s_full, beta_full=beta_full)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    lp_all = np.log(prices)
    tr_all, te_all = split(lp_all)
    summary = {}

    # ---------------------------------------------------------- 1. pair selection
    rows = []
    for x, y in CANDIDATE_PAIRS:
        r_tr = tr_all[[x, y]].diff().dropna()
        _, p_tr, _ = coint(tr_all[x], tr_all[y])
        _, p_te, _ = coint(te_all[x], te_all[y])
        rows.append({"pair": f"{x}/{y}", "return_corr_train": r_tr[x].corr(r_tr[y]),
                     "eg_pvalue_train": p_tr, "eg_pvalue_test": p_te,
                     "passes_5pct": p_tr < 0.05, "passes_bonferroni": p_tr < 0.05 / len(CANDIDATE_PAIRS)})
    sel = pd.DataFrame(rows).set_index("pair")
    sel.round(4).to_csv(TAB / "pair_selection.csv")
    print(sel.round(4).to_string())

    # --------------------------------------------------------------- 2. V / MA
    print("running V/MA ...")
    vm = run_pair(prices, "MA", "V")
    met = pd.DataFrame({n: B.metrics(vm["results"][n]) for n in ORDER}).T
    cis = {n: B.sharpe_ci(vm["results"][n]["ret"]) if met.loc[n, "Trades"] > 0 else (np.nan,) * 3 for n in ORDER}
    met["Sharpe 95% CI low"] = [cis[n][0] for n in ORDER]
    met["Sharpe 95% CI high"] = [cis[n][1] for n in ORDER]
    met["P(Sharpe<=0)"] = [cis[n][2] for n in ORDER]
    met.to_csv(TAB / "vma_results.csv")
    print(met.round(3).to_string())
    vm["grid"].to_csv(TAB / "latent_qbeta_profile.csv", index=False)
    print(vm["grid"].round(4).to_string())

    # Diagnosis of the original Kalman filter
    so = vm["sigs"][2]
    resid_ols = sm.OLS(vm["tr"]["MA"], sm.add_constant(vm["tr"]["V"])).fit().resid
    diag = {"R_original": so.extra["R"], "R_mle": vm["mle"]["R"],
            "R_ratio": so.extra["R"] / vm["mle"]["R"],
            "innovation_sd_test": so.extra["innovation_sd"], "mean_sqrt_S_test": so.extra["sqrt_S"],
            "typical_abs_z": float(so.z.abs().median()), "max_abs_z_test": float(so.z.abs().max()),
            "ols_spread_daily_change_sd": float(resid_ols.diff().std()),
            "ols_spread_level_sd": float(resid_ols.std()),
            "ols_spread_lag1_autocorr": float(resid_ols.autocorr(1)),
            "mle_q_alpha": vm["mle"]["q_alpha"], "mle_q_beta": vm["mle"]["q_beta"],
            "latent_free_mle": vm["free"]}
    summary["diagnosis"] = diag
    print(json.dumps(diag, indent=1, default=float))

    # Did the relationship survive out of sample?
    st = vm["sigs"][0]
    test_spread = vm["te"]["MA"] - st.extra["alpha"] - st.extra["beta"] * vm["te"]["V"]
    summary["test_period"] = {"adf_p_static_spread_test": float(adfuller(test_spread)[1]),
                              "static_z_min": float(st.z.min()), "static_z_max": float(st.z.max()),
                              "static_z_mean": float(st.z.mean())}

    # ------------------------------------------------------- 3. robustness grid
    rows = []
    for s in vm["sigs"]:
        for entry in (1.5, 2.0, 2.5):
            for cost in (0, 5, 10, 20):
                m = B.metrics(B.run(s.z, s.beta, vm["rets"], entry, cost))
                rows.append({"model": s.name, "entry": entry, "cost_bps": cost, "sharpe": m["Sharpe"],
                             "annual_return": m["Annual return"], "trades": m["Trades"]})
    rob = pd.DataFrame(rows)
    rob.to_csv(TAB / "robustness_entry_cost.csv", index=False)

    # Calendar-year returns
    yr = pd.DataFrame({n: (1 + vm["results"][n]["ret"]).groupby(vm["results"][n]["ret"].index.year).prod() - 1
                       for n in ORDER})
    yr.to_csv(TAB / "vma_returns_by_year.csv")

    # --------------------------------------------------------- 4. other pairs
    print("running other pairs (generalization check) ...")
    others = {}
    for x, y in CANDIDATE_PAIRS:
        if (x, y) == ("V", "MA"):
            continue
        res = run_pair(prices, y, x)
        others[f"{x}/{y}"] = {n: B.metrics(res["results"][n]) for n in ORDER}
    oth = pd.concat({k: pd.DataFrame(v).T for k, v in others.items()}, names=["pair", "model"])
    oth = pd.concat([pd.concat({"V/MA": met[oth.columns]}, names=["pair", "model"]), oth])
    oth[["Annual return", "Sharpe", "Max drawdown", "Trades"]].to_csv(TAB / "all_pairs_results.csv")
    print(oth[["Annual return", "Sharpe", "Max drawdown", "Trades"]].round(3).to_string())

    # --------------------------------------------------------------- 5. figures
    P.fig_pair(prices, tr_all, FIG / "01_pair_selection.png", sel)
    P.fig_hedge_ratios(vm, FIG / "02_hedge_ratios.png")
    P.fig_diagnosis(vm, FIG / "03_kalman_diagnosis.png")
    P.fig_frontier(vm["grid"], FIG / "04_adaptivity_frontier.png")
    P.fig_equity(vm, ORDER, FIG / "05_equity_curves.png")
    P.fig_sharpe_ci(met, FIG / "06_sharpe_uncertainty.png")
    P.fig_latent_signal(vm, FIG / "07_latent_spread_signal.png")
    P.fig_all_pairs(oth, ORDER, FIG / "08_all_pairs.png")

    # ---------------------------------------------------------------- 6. summary
    summary["pair_selection"] = sel.round(4).reset_index().to_dict(orient="records")
    summary["vma"] = met.round(4).reset_index().rename(columns={"index": "model"}).to_dict(orient="records")
    summary["latent_choice"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in vm["sigs"][5].extra.items()}
    summary["kalman_mle"] = vm["mle"]
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("done")


if __name__ == "__main__":
    main()
