"""Figures for the README and research report (matplotlib, print-friendly light theme)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"

COLORS = {"Static OLS": MUTED, "Rolling OLS": ORANGE, "Kalman (original spec)": RED,
          "Kalman (MLE) innovation": MAGENTA, "Kalman beta + rolling z": AQUA,
          "Latent-spread state space": BLUE}
SHORT = {"Static OLS": "Static OLS", "Rolling OLS": "Rolling OLS (126d)",
         "Kalman (original spec)": "Kalman, original spec", "Kalman (MLE) innovation": "Kalman, MLE noise",
         "Kalman beta + rolling z": "Kalman β + rolling z", "Latent-spread state space": "Latent-spread model"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2, "font.size": 9.5, "lines.linewidth": 1.4,
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _shade_test(ax, te_start):
    ax.axvspan(te_start, ax.get_xlim()[1] if False else pd.Timestamp("2025-12-31"), color="#f3f2ee", zorder=0)


def fig_pair(prices, tr, path, sel):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    norm = prices[["V", "MA"]] / prices[["V", "MA"]].iloc[0] * 100
    ax = axes[0]
    ax.axvspan(pd.Timestamp("2022-01-01"), norm.index[-1], color="#f3f2ee", zorder=0)
    ax.plot(norm.index, norm["V"], color=BLUE, label="Visa (V)")
    ax.plot(norm.index, norm["MA"], color=ORANGE, label="Mastercard (MA)")
    ax.text(pd.Timestamp("2022-02-01"), norm.max().max() * 0.97, "test period", color=INK2, fontsize=8.5)
    ax.set_title("V and MA, normalized to 100 (2016-2025)")
    ax.legend(loc="upper left")
    ax = axes[1]
    s = sel.sort_values("eg_pvalue_train", ascending=False)
    ax.barh(s.index, s["eg_pvalue_train"], color=[BLUE if v < 0.05 else MUTED for v in s["eg_pvalue_train"]], height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1.5)
    ax.axvline(0.05, color=INK2, ls="--", lw=0.9)
    ax.axvline(0.0125, color=RED, ls=":", lw=0.9)
    for i, v in enumerate(s["eg_pvalue_train"]):
        ax.text(v * 1.12, i, f"p = {v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.text(0.052, len(s) - 0.45, "5%", color=INK2, fontsize=8)
    ax.text(0.0115, len(s) - 0.45, "Bonferroni", color=RED, fontsize=8, ha="right")
    ax.set_ylim(-0.6, len(s) - 0.2)
    ax.set_title("Engle-Granger p-value, training data (log scale)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def fig_hedge_ratios(vm, path):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    te0 = vm["te"].index[0]
    ax.axvspan(te0, vm["te"].index[-1], color="#f3f2ee", zorder=0)
    st, ro, ko, km, kb, la = vm["sigs"]
    ax.axhline(st.extra["beta"], color=MUTED, ls="--", lw=1.1, label=f"Static OLS ({st.extra['beta']:.3f})")
    ax.plot(ro.beta.index, ro.beta, color=ORANGE, lw=1.0, label="Rolling OLS (126d)")
    ax.plot(kb.beta.index, kb.beta, color=AQUA, lw=1.3, label="Kalman, MLE noise")
    ax.plot(ko.beta.index, ko.beta, color=RED, lw=1.3, label="Kalman, original spec")
    ax.plot(vm["beta_full"].loc[te0:].index, vm["beta_full"].loc[te0:], color=BLUE, lw=1.6, label="Latent-spread model")
    ax.set_ylabel("hedge ratio β (log MA on log V)")
    ax.set_title("Out-of-sample hedge ratios: how fast each estimator lets β move")
    ax.set_xlim(te0 - pd.Timedelta(days=10), vm["te"].index[-1])
    ax.legend(loc="upper left", ncols=3, fontsize=8.5)
    _save(fig, path)


def fig_diagnosis(vm, path):
    st, ro, ko, km, kb, la = vm["sigs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    ax.plot(st.z.index, st.z, color=MUTED, lw=1.0, label="Static OLS z-score")
    ax.plot(ko.z.index, ko.z, color=RED, lw=1.0, label="Original Kalman z = e / √S")
    for v in (2, -2):
        ax.axhline(v, color=INK2, ls="--", lw=0.8)
    ax.set_title("Test-period signals: the original Kalman z never reaches ±2")
    ax.legend(loc="lower left", fontsize=8.5)
    ax = axes[1]
    e_sd = ko.extra["innovation_sd"]
    vals = [e_sd, ko.extra["sqrt_S"]]
    bars = ax.bar(["Daily forecast\nerror sd", "Filter's assumed\nerror sd (√S)"], vals, color=[INK2, RED], width=0.55)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Why: R was set ~200× too large")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def fig_frontier(grid, path):
    g = grid.copy()
    x = np.arange(len(g))
    labels = ["0" if q == 0 else f"{q:.0e}".replace("e-0", "e-") for q in g["q_beta"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    axes[0].plot(x, g["loglik"] - g["loglik"].min(), "o-", color=INK2)
    axes[0].set_title("Likelihood gain")
    axes[0].set_ylabel("Δ log-likelihood (train)")
    axes[1].plot(x, g["half_life_days"], "o-", color=BLUE)
    axes[1].set_yscale("log")
    axes[1].set_title("Spread half-life (days)")
    axes[2].plot(x, g["train_sharpe"], "o-", color=ORANGE, label="train (selection)")
    axes[2].plot(x, g["test_sharpe"], "s--", color=AQUA, label="test (not used)")
    axes[2].axhline(0, color=INK2, lw=0.8)
    axes[2].set_title("Strategy Sharpe")
    axes[2].legend(fontsize=8)
    pick = int(g["train_sharpe"].fillna(-np.inf).idxmax())
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_xlabel("hedge-ratio drift q_β")
        ax.axvline(pick, color=MUTED, ls=":", lw=1)
    fig.suptitle("The adaptivity trade-off: a faster-moving β fits better but leaves less to trade",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    _save(fig, path)


def fig_equity(vm, order, path):
    fig, ax = plt.subplots(figsize=(10, 3.9))
    for n in order:
        eq = (1 + vm["results"][n]["ret"]).cumprod()
        lw = 2.0 if n == "Latent-spread state space" else 1.3
        ax.plot(eq.index, eq, color=COLORS[n], lw=lw, label=SHORT[n])
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_ylabel("growth of $1")
    ax.set_title("Out-of-sample equity, 2022-2025 (5 bp costs)")
    ax.legend(loc="upper left", ncols=3, fontsize=8.5)
    _save(fig, path)


def fig_sharpe_ci(met, path):
    m = met[met["Trades"] > 0]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    y = np.arange(len(m))[::-1]
    for yi, (n, r) in zip(y, m.iterrows()):
        ax.plot([r["Sharpe 95% CI low"], r["Sharpe 95% CI high"]], [yi, yi], color=COLORS[n], lw=2.4, solid_capstyle="round")
        ax.plot(r["Sharpe"], yi, "o", color=COLORS[n], ms=8, mec="white", mew=1.5)
        ax.text(r["Sharpe 95% CI high"] + 0.08, yi, f"{r['Sharpe']:.2f}  ({int(r['Trades'])} trades)", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[n] for n in m.index])
    ax.set_xlabel("annualized Sharpe, test period (dot), 95% block-bootstrap interval; few trades make intervals unreliable")
    ax.set_title("Wide intervals: four years of trading cannot separate these strategies")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def fig_latent_signal(vm, path):
    la = vm["sigs"][5]
    pos = vm["results"]["Latent-spread state space"]["pos"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(la.z.index, la.z, color=BLUE, lw=1.0, label="filtered spread s_t|t / sd")
    for v in (2, -2):
        ax.axhline(v, color=INK2, ls="--", lw=0.8)
    ax.axhline(0, color=INK2, lw=0.6)
    if (pos > 0).any():
        ax.fill_between(pos.index, -3.5, 3.5, where=pos > 0, color=AQUA, alpha=0.15, lw=0, label="long spread")
    ax.fill_between(pos.index, -3.5, 3.5, where=pos < 0, color=ORANGE, alpha=0.15, lw=0, label="short spread (MA rich vs V)")
    ax.set_ylim(-3.5, 3.5)
    ax.set_title("Latent-spread model: the signal and the positions it took, 2022-2025")
    ax.legend(loc="lower left", ncols=3, fontsize=8.5)
    _save(fig, path)


def fig_all_pairs(oth, order, path):
    pairs = list(dict.fromkeys(oth.index.get_level_values(0)))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.13
    x = np.arange(len(pairs))
    for i, n in enumerate(order):
        vals = [oth.loc[(p, n), "Sharpe"] if (p, n) in oth.index else np.nan for p in pairs]
        vals = [0 if np.isnan(v) else v for v in vals]
        ax.bar(x + (i - 2.5) * w, vals, width=w * 0.92, color=COLORS[n], label=SHORT[n])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel("test-period Sharpe")
    ax.set_title("Same six strategies on all four candidate pairs (only V/MA passed the cointegration screen)")
    ax.legend(ncols=6, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), handlelength=1.2, columnspacing=1.0)
    ax.grid(axis="x", visible=False)
    _save(fig, path)
