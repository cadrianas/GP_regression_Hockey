"""
GP Regression: Within-Period xGoal Dynamics in Comeback Situations
===================================================================
REVISED: Accounts for era heterogeneity (2014-20 vs 2022-24)

Core insight from decomposition:
  - 2014-20 (early era): mean xGoal ~0.062 per shot
  - 2022-24 (recent era): mean xGoal ~0.072 per shot (+16% shift)

This script fits TWO GPs:
  1. Early era (2014-20): Baseline temporal dynamics
  2. Recent era (2022-24): Elevated temporal dynamics

Then validates both against 2024-25 to test whether trend continues.

Question: Are the shapes the same but levels different?
Or did the temporal dynamics themselves change?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.append(str(ROOT))

from src.paths import DATA_DIR, RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ── Configuration ─────────────────────────────────────────────────────────────

SEASONS = {
    DATA_DIR / "shots_2014.csv": {"label": "2014-15", "era": "early"},
    DATA_DIR / "shots_2015.csv": {"label": "2015-16", "era": "early"},
    DATA_DIR / "shots_2016.csv": {"label": "2016-17", "era": "early"},
    DATA_DIR / "shots_2017.csv": {"label": "2017-18", "era": "early"},
    DATA_DIR / "shots_2018.csv": {"label": "2018-19", "era": "early"},
    DATA_DIR / "shots_2019.csv": {"label": "2019-20", "era": "early"},
    DATA_DIR / "shots_2020.csv": {"label": "2020-21", "era": "early"},   # Include COVID for continuity
    DATA_DIR / "shots_2021.csv": {"label": "2021-22", "era": "early"},
    DATA_DIR / "shots_2022.csv": {"label": "2022-23", "era": "recent"},
    DATA_DIR / "shots_2023.csv": {"label": "2023-24", "era": "recent"},
    DATA_DIR / "shots_2024.csv": {"label": "2024-25", "era": "validate"},
}

PULL_CUTOFF = 1110
BIN_SIZE    = 60
N_GRID      = 500

# ── Load & Filter ─────────────────────────────────────────────────────────────

def load_and_filter(filepath, season_label):
    df = pd.read_csv(filepath)
    df["period"]        = df["period"].astype(int)
    df["homeTeamGoals"] = df["homeTeamGoals"].astype(int)
    df["awayTeamGoals"] = df["awayTeamGoals"].astype(int)

    mask = (
        (df["period"] == 3) &
        (abs(df["homeTeamGoals"] - df["awayTeamGoals"]) == 2)
    )
    df3 = df[mask].copy()

    df3["trailingTeam"] = np.where(
        df3["homeTeamGoals"] < df3["awayTeamGoals"],
        df3["homeTeamCode"], df3["awayTeamCode"]
    ).astype(str)
    df3 = df3[df3["teamCode"].astype(str) == df3["trailingTeam"]].copy()

    df3["time_in_period"] = df3["time"] - 2400
    df3 = df3[df3["time_in_period"] <= PULL_CUTOFF].copy()
    df3["season"] = season_label
    return df3


print("Loading data by era...\n")
all_dfs = []
era_counts = {"early": 0, "recent": 0, "validate": 0}

for filepath, meta in SEASONS.items():
    if not filepath.exists():
        continue
    df_s = load_and_filter(filepath, meta["label"])
    df_s["era"] = meta["era"]
    all_dfs.append(df_s)
    era_counts[meta["era"]] += len(df_s)
    print(f"  {meta['label']:12s} ({meta['era']:8s}): {len(df_s):5,} shots")

full = pd.concat(all_dfs, ignore_index=True)

data_early    = full[full["era"] == "early"].copy()
data_recent   = full[full["era"] == "recent"].copy()
data_validate = full[full["era"] == "validate"].copy()

print(f"\nEra breakdown:")
print(f"  Early (2014-21):    {len(data_early):6,} shots")
print(f"  Recent (2022-24):   {len(data_recent):6,} shots")
print(f"  Validate (2024-25): {len(data_validate):6,} shots")

# ── Binning Function ──────────────────────────────────────────────────────────

bin_edges = np.arange(0, PULL_CUTOFF + BIN_SIZE, BIN_SIZE)

def make_bins(df, label):
    df = df.copy()
    df["bin"] = pd.cut(
        df["time_in_period"],
        bins=bin_edges,
        right=False,
        include_lowest=True,
    )
    agg = (
        df.groupby("bin", observed=True)["xGoal"]
        .agg(n="count", mean_xG="mean", var_xG="var")
        .reset_index()
    )
    agg["midpoint"]  = agg["bin"].apply(lambda b: (b.left + b.right) / 2).astype(float)
    agg["se_sq"]     = agg["var_xG"] / agg["n"]
    agg["label"]     = label
    agg = agg[agg["n"] >= 10].copy()  # Drop noisy bins
    return agg.reset_index(drop=True)


bins_early    = make_bins(data_early, "early")
bins_recent   = make_bins(data_recent, "recent")
bins_validate = make_bins(data_validate, "validate")

print(f"\nBinned data:")
print(f"  Early bins:    {len(bins_early):2d}  |  mean xGoal = {bins_early['mean_xG'].mean():.5f}")
print(f"  Recent bins:   {len(bins_recent):2d}  |  mean xGoal = {bins_recent['mean_xG'].mean():.5f}")
print(f"  Validate bins: {len(bins_validate):2d}  |  mean xGoal = {bins_validate['mean_xG'].mean():.5f}")

# ── Fit Two GPs ───────────────────────────────────────────────────────────────

def fit_gp(bins_df, label):
    """Fit GP to binned data."""
    X = bins_df["midpoint"].values.reshape(-1, 1)
    y = bins_df["mean_xG"].values
    alpha = bins_df["se_sq"].values

    kernel = (
        ConstantKernel(constant_value=0.01, constant_value_bounds=(1e-4, 10.0)) *
        Matern(length_scale=200.0, length_scale_bounds=(30.0, 1000.0), nu=2.5) +
        WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1.0))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42,
    )

    print(f"\nFitting GP for {label} era...")
    gp.fit(X, y)
    print(f"  Kernel: {gp.kernel_}")
    print(f"  Log marginal likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")

    return gp


gp_early  = fit_gp(bins_early, "early")
gp_recent = fit_gp(bins_recent, "recent")

# ── Predict on Fine Grid ──────────────────────────────────────────────────────

t_grid = np.linspace(0, PULL_CUTOFF, N_GRID).reshape(-1, 1)

mu_early, std_early = gp_early.predict(t_grid, return_std=True)
ci_lo_early = mu_early - 1.96 * std_early
ci_hi_early = mu_early + 1.96 * std_early

mu_recent, std_recent = gp_recent.predict(t_grid, return_std=True)
ci_lo_recent = mu_recent - 1.96 * std_recent
ci_hi_recent = mu_recent + 1.96 * std_recent

gp_early_results = pd.DataFrame({
    "time_s": t_grid.flatten(),
    "mu": mu_early,
    "std": std_early,
    "ci_lo": ci_lo_early,
    "ci_hi": ci_hi_early,
})
gp_early_results.to_csv(RESULTS_DIR / "gp_results_early.csv", index=False, float_format="%.6f")

gp_recent_results = pd.DataFrame({
    "time_s": t_grid.flatten(),
    "mu": mu_recent,
    "std": std_recent,
    "ci_lo": ci_lo_recent,
    "ci_hi": ci_hi_recent,
})
gp_recent_results.to_csv(RESULTS_DIR / "gp_results_recent.csv", index=False, float_format="%.6f")

print("\nSaved: gp_results_early.csv, gp_results_recent.csv")

# ── Validation Check Against Both GPs ─────────────────────────────────────────

print("\n" + "="*70)
print("Validation: 2024-25 vs Early Era GP")
print("="*70)

X_validate = bins_validate["midpoint"].values.reshape(-1, 1)
mu_v_vs_early, std_v_vs_early = gp_early.predict(X_validate, return_std=True)
ci_lo_v_early = mu_v_vs_early - 1.96 * std_v_vs_early
ci_hi_v_early = mu_v_vs_early + 1.96 * std_v_vs_early
inside_early = (bins_validate["mean_xG"].values >= ci_lo_v_early) & \
               (bins_validate["mean_xG"].values <= ci_hi_v_early)

print(f"  Validation bins inside early-era GP CI: {inside_early.sum()} / {len(inside_early)}")
if inside_early.sum() < len(inside_early):
    print(f"  → {(~inside_early).sum()} bins OUTSIDE early GP  [evidence of strategic shift]")

print("\n" + "="*70)
print("Validation: 2024-25 vs Recent Era GP")
print("="*70)

mu_v_vs_recent, std_v_vs_recent = gp_recent.predict(X_validate, return_std=True)
ci_lo_v_recent = mu_v_vs_recent - 1.96 * std_v_vs_recent
ci_hi_v_recent = mu_v_vs_recent + 1.96 * std_v_vs_recent
inside_recent = (bins_validate["mean_xG"].values >= ci_lo_v_recent) & \
                (bins_validate["mean_xG"].values <= ci_hi_v_recent)

print(f"  Validation bins inside recent-era GP CI: {inside_recent.sum()} / {len(inside_recent)}")
if inside_recent.sum() == len(inside_recent):
    print(f"  → ALL bins inside recent GP  [trend is stable/continuing]")

# ── Figures ───────────────────────────────────────────────────────────────────

t_flat = t_grid.flatten()

# ── Fig 6a: Early Era GP ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

ax.fill_between(t_flat, ci_lo_early, ci_hi_early, alpha=0.25, color="#2166ac",
                label="Early era GP 95% CI")
ax.plot(t_flat, mu_early, color="#2166ac", linewidth=2.5, label="Early era GP mean")
ax.scatter(bins_early["midpoint"], bins_early["mean_xG"],
           s=bins_early["n"] / bins_early["n"].max() * 120,
           color="#d73027", zorder=5, label="Training bins (2014-21)")
ax.errorbar(bins_early["midpoint"], bins_early["mean_xG"],
            yerr=1.96 * np.sqrt(bins_early["se_sq"]),
            fmt="none", color="#d73027", linewidth=1, alpha=0.5)

# Overlay validation points
ax.scatter(bins_validate["midpoint"], bins_validate["mean_xG"],
           s=70, color="#4dac26", marker="^", zorder=6,
           label=f"Validation 2024-25\n[{inside_early.sum()}/{len(inside_early)} inside CI]")

for boundary in [300, 600, 900]:
    ax.axvline(boundary, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)

ax.set_xlabel("Time into third period (seconds)", fontsize=11)
ax.set_ylabel("Mean xGoal per shot", fontsize=11)
ax.set_title("Early Era (2014-21) GP Posterior\nwith Recent Validation Data Overlaid\n"
             "(Points outside CI = evidence of strategic shift)", fontsize=11)
ax.set_xlim(0, PULL_CUTOFF)
ax.legend(fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6a_gp_early.png", bbox_inches="tight")
plt.close()
print("Saved: fig6a_gp_early.png")

# ── Fig 6b: Recent Era GP ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

ax.fill_between(t_flat, ci_lo_recent, ci_hi_recent, alpha=0.25, color="#d73027",
                label="Recent era GP 95% CI")
ax.plot(t_flat, mu_recent, color="#d73027", linewidth=2.5, label="Recent era GP mean")
ax.scatter(bins_recent["midpoint"], bins_recent["mean_xG"],
           s=bins_recent["n"] / bins_recent["n"].max() * 120,
           color="#2166ac", zorder=5, label="Training bins (2022-24)")
ax.errorbar(bins_recent["midpoint"], bins_recent["mean_xG"],
            yerr=1.96 * np.sqrt(bins_recent["se_sq"]),
            fmt="none", color="#2166ac", linewidth=1, alpha=0.5)

# Overlay validation points
ax.scatter(bins_validate["midpoint"], bins_validate["mean_xG"],
           s=70, color="#4dac26", marker="^", zorder=6,
           label=f"Validation 2024-25\n[{inside_recent.sum()}/{len(inside_recent)} inside CI]")

for boundary in [300, 600, 900]:
    ax.axvline(boundary, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)

ax.set_xlabel("Time into third period (seconds)", fontsize=11)
ax.set_ylabel("Mean xGoal per shot", fontsize=11)
ax.set_title("Recent Era (2022-24) GP Posterior\nwith Validation Data (2024-25)\n"
             "(All points inside CI = trend continues)", fontsize=11)
ax.set_xlim(0, PULL_CUTOFF)
ax.legend(fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6b_gp_recent.png", bbox_inches="tight")
plt.close()
print("Saved: fig6b_gp_recent.png")

# ── Fig 7: Overlay Both GPs ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(t_flat, ci_lo_early, ci_hi_early, alpha=0.15, color="#2166ac",
                label="Early era 95% CI")
ax.plot(t_flat, mu_early, color="#2166ac", linewidth=2.5, label="Early era mean (2014-21)")

ax.fill_between(t_flat, ci_lo_recent, ci_hi_recent, alpha=0.15, color="#d73027",
                label="Recent era 95% CI")
ax.plot(t_flat, mu_recent, color="#d73027", linewidth=2.5, label="Recent era mean (2022-24)")

# Add all bin points
ax.scatter(bins_early["midpoint"], bins_early["mean_xG"],
           s=50, color="#2166ac", alpha=0.6, zorder=4)
ax.scatter(bins_recent["midpoint"], bins_recent["mean_xG"],
           s=50, color="#d73027", alpha=0.6, zorder=4)

# Validation overlay
ax.scatter(bins_validate["midpoint"], bins_validate["mean_xG"],
           s=80, color="#4dac26", marker="^", zorder=6,
           label="Validation 2024-25", edgecolor="black", linewidth=0.5)

for boundary in [300, 600, 900]:
    ax.axvline(boundary, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)

# Shade the difference
ax.fill_between(t_flat, mu_early, mu_recent, alpha=0.15, color="orange",
                label="Era difference in posterior mean")

ax.set_xlabel("Time into third period (seconds)", fontsize=11)
ax.set_ylabel("Mean xGoal per shot", fontsize=11)
ax.set_title("Comparison: Early vs Recent Era GPs\n"
             "Strategic Shift Visualized as Level Difference", fontsize=12, fontweight="bold")
ax.set_xlim(0, PULL_CUTOFF)
ax.legend(fontsize=10, framealpha=0.95, loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig7_gp_comparison.png", bbox_inches="tight")
plt.close()
print("Saved: fig7_gp_comparison.png")

# ── Summary Statistics ────────────────────────────────────────────────────────

print("\n" + "="*70)
print("SUMMARY: Mean xGoal Across Period")
print("="*70)

print(f"\nEarly era (2014-21):")
print(f"  GP posterior mean: {mu_early.mean():.5f}")
print(f"  Training bin mean: {bins_early['mean_xG'].mean():.5f}")

print(f"\nRecent era (2022-24):")
print(f"  GP posterior mean: {mu_recent.mean():.5f}")
print(f"  Training bin mean: {bins_recent['mean_xG'].mean():.5f}")

print(f"\nValidation (2024-25):")
print(f"  Observed mean: {bins_validate['mean_xG'].mean():.5f}")

print(f"\nEra shift:")
shift = (mu_recent.mean() - mu_early.mean()) / mu_early.mean() * 100
print(f"  Recent vs Early: {shift:+.1f}%")

print("\nDone.")