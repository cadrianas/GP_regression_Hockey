"""
GP Regression: Within-Period xGoal Dynamics in Comeback Situations
===================================================================
Follows decomposition_analysis.py — run that script first to confirm
your data loads correctly, then run this script.

Training:   seasons 2014-2015 to 2023-2024
Validation:  season 2024-25

What this script produces
-------------------------
  gp_bin_data_train.csv       Binned training data fed to the GP
  gp_bin_data_valid.csv       Binned validation data
  gp_results.csv              GP posterior mean + CI at fine grid
  exponential_test.csv        Exponential fit vs GP CI, pointwise
  fig6_gp_posterior.png       GP posterior + training bin means
  fig7_exponential_overlay.png GP posterior + exponential fit
  fig8_validation_gp.png      Validation bins vs training posterior
  fig9_semilog.png            Semi-log plot (exponential appears linear if fit holds)

Dependencies: pandas, numpy, matplotlib, scipy, scikit-learn
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

# Find project root dynamically
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

# ── 0. Configuration ──────────────────────────────────────────────────────────

SEASONS = {
    DATA_DIR / "shots_2014.csv": {"label": "2014-15", "role": "train"},
    DATA_DIR / "shots_2015.csv": {"label": "2015-16", "role": "train"},
    DATA_DIR / "shots_2016.csv": {"label": "2016-17", "role": "train"},
    DATA_DIR / "shots_2017.csv": {"label": "2017-18", "role": "train"},
    DATA_DIR / "shots_2018.csv": {"label": "2018-19", "role": "train"},
    DATA_DIR / "shots_2019.csv": {"label": "2019-20", "role": "train"},
    DATA_DIR / "shots_2020.csv": {"label": "2020-21", "role": "train"},
    DATA_DIR / "shots_2021.csv": {"label": "2021-22", "role": "train"},
    DATA_DIR / "shots_2022.csv": {"label": "2022-23", "role": "train"},
    DATA_DIR / "shots_2023.csv": {"label": "2023-24", "role": "train"},
    DATA_DIR / "shots_2024.csv": {"label": "2024-25", "role": "validate"},
}

PULL_CUTOFF = 1110      # seconds into period; exclude goalie-pull epoch
BIN_SIZE    = 60        # seconds per bin → 18 bins across 0–1110 s
N_GRID      = 500       # points in the fine prediction grid

# ── 1. Load & Filter (same logic as decomposition script) ────────────────────

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


print("Loading data...")
all_dfs = []
for filepath, meta in SEASONS.items():
    if not filepath.exists():
        print(f"  SKIPPING {filepath} — not found")
        continue
    df_s = load_and_filter(filepath, meta["label"])
    df_s["role"] = meta["role"]
    all_dfs.append(df_s)
    print(f"  {meta['label']}: {len(df_s):,} shots")

full  = pd.concat(all_dfs, ignore_index=True)
train = full[full["role"] == "train"].copy()
valid = full[full["role"] == "validate"].copy()
print(f"\nTraining: {len(train):,} shots | Validation: {len(valid):,} shots")

# ── 2. Bin into 60-Second Windows ────────────────────────────────────────────
# Each bin's mean xGoal is the GP observation.
# Noise level alpha_i = variance_i / n_i (standard error squared).
# This gives the GP heteroscedastic noise — bins with fewer shots
# are treated as less certain observations.

bin_edges = np.arange(0, PULL_CUTOFF + BIN_SIZE, BIN_SIZE)


def make_bins(df, label):
    df = df.copy()
    df["bin"] = pd.cut(
        df["time_in_period"],
        bins=bin_edges,
        right=False,           # [left, right)
        include_lowest=True,
    )
    agg = (
        df.groupby("bin", observed=True)["xGoal"]
        .agg(n="count", mean_xG="mean", var_xG="var")
        .reset_index()
    )
    agg["midpoint"]  = agg["bin"].apply(lambda b: (b.left + b.right) / 2).astype(float)
    agg["se_sq"]     = agg["var_xG"] / agg["n"]   # variance of the mean
    agg["label"]     = label
    # Drop bins with fewer than 10 shots (too noisy to be useful)
    agg = agg[agg["n"] >= 10].copy()
    return agg.reset_index(drop=True)


bins_train = make_bins(train, "training")
bins_valid = make_bins(valid, "validation")

print(f"\nTraining bins: {len(bins_train)}  |  Validation bins: {len(bins_valid)}")
print(bins_train[["midpoint", "n", "mean_xG", "se_sq"]].to_string(index=False,
      float_format="{:.5f}".format))

bins_train.to_csv(RESULTS_DIR / "gp_bin_data_train.csv", index=False, float_format="%.6f")
bins_valid.to_csv(RESULTS_DIR / "gp_bin_data_valid.csv", index=False, float_format="%.6f")

# ── 3. Fit GP ─────────────────────────────────────────────────────────────────
#
# Kernel: ConstantKernel × Matérn(nu=2.5) + WhiteKernel
#
# ConstantKernel scales the overall amplitude of the GP.
# Matérn(nu=2.5) is twice-differentiable — appropriate for a smooth
# but non-analytic function. Rougher than RBF (infinitely smooth),
# which is important because we don't want to over-smooth the window-4 spike.
# WhiteKernel absorbs noise not captured by the heteroscedastic alpha.
#
# alpha: per-observation noise variance (our se_sq estimates).
# normalize_y: centers the target on its mean, improving numerical stability.

X_train = bins_train["midpoint"].values.reshape(-1, 1)
y_train = bins_train["mean_xG"].values
alpha   = bins_train["se_sq"].values   # heteroscedastic noise

kernel = (
    ConstantKernel(constant_value=0.01, constant_value_bounds=(1e-4, 10.0)) *
    Matern(length_scale=200.0, length_scale_bounds=(30.0, 1000.0), nu=2.5) +
    WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1.0))
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha,
    n_restarts_optimizer=10,   # avoid local optima in marginal likelihood
    normalize_y=True,
    random_state=42,
)

print("\nFitting GP (this may take a few seconds)...")
gp.fit(X_train, y_train)
print(f"Optimised kernel: {gp.kernel_}")
print(f"Log marginal likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")

# ── 4. Predict on Fine Grid ───────────────────────────────────────────────────

t_grid  = np.linspace(0, PULL_CUTOFF, N_GRID).reshape(-1, 1)
mu, std = gp.predict(t_grid, return_std=True)
ci_lo   = mu - 1.96 * std
ci_hi   = mu + 1.96 * std

gp_results = pd.DataFrame({
    "time_s":  t_grid.flatten(),
    "mu":      mu,
    "std":     std,
    "ci_lo":   ci_lo,
    "ci_hi":   ci_hi,
})
gp_results.to_csv(RESULTS_DIR / "gp_results.csv", index=False, float_format="%.6f")
print("Saved: gp_results.csv")

# ── 5. Exponential Fit & Consistency Test ────────────────────────────────────
# Fit y = a * exp(b * t) to the bin means.
# Then check pointwise whether the exponential falls inside the GP 95% CI.
# We report the fraction of grid points where it falls outside.

def exponential(t, a, b):
    return a * np.exp(b * t)

try:
    popt, pcov = curve_fit(
        exponential,
        bins_train["midpoint"].values,
        bins_train["mean_xG"].values,
        p0=[0.06, 0.0005],
        maxfev=5000,
    )
    a_fit, b_fit = popt
    print(f"\nExponential fit: a={a_fit:.5f}, b={b_fit:.6f}")
    print(f"  Implied doubling time: {np.log(2)/b_fit:.1f} seconds "
          f"({np.log(2)/b_fit/60:.1f} minutes)")

    exp_pred = exponential(t_grid.flatten(), a_fit, b_fit)

    # Pointwise consistency: is exponential within GP 95% CI?
    inside      = (exp_pred >= ci_lo) & (exp_pred <= ci_hi)
    pct_inside  = inside.mean() * 100

    print(f"  Exponential inside GP 95% CI: {pct_inside:.1f}% of grid points")

    # Also check at the fine-grid level where it exits
    if pct_inside < 100:
        outside_times = t_grid.flatten()[~inside]
        print(f"  Outside CI at times (s): {outside_times[0]:.0f}–{outside_times[-1]:.0f}")

    exp_test = pd.DataFrame({
        "time_s":      t_grid.flatten(),
        "exp_pred":    exp_pred,
        "gp_mu":       mu,
        "gp_ci_lo":    ci_lo,
        "gp_ci_hi":    ci_hi,
        "inside_ci":   inside,
    })
    exp_test.to_csv(RESULTS_DIR / "exponential_test.csv", index=False, float_format="%.6f")
    print("Saved: exponential_test.csv")
    exp_fitted = True

except RuntimeError as e:
    print(f"  Exponential fit failed: {e}")
    exp_fitted = False

# ── 6. Validation Posterior Predictive Check ──────────────────────────────────
# Predict the GP posterior at validation bin midpoints and check whether
# the validation means fall within the posterior predictive interval.

X_valid    = bins_valid["midpoint"].values.reshape(-1, 1)
mu_v, std_v = gp.predict(X_valid, return_std=True)
ci_lo_v    = mu_v - 1.96 * std_v
ci_hi_v    = mu_v + 1.96 * std_v
inside_v   = (bins_valid["mean_xG"].values >= ci_lo_v) & \
             (bins_valid["mean_xG"].values <= ci_hi_v)

print(f"\nValidation posterior predictive check:")
print(f"  Bins inside GP 95% CI: {inside_v.sum()} / {len(inside_v)} "
      f"({inside_v.mean()*100:.0f}%)")

# ── 7. Figures ────────────────────────────────────────────────────────────────

t_flat = t_grid.flatten()

# ── Fig 6: GP posterior + training bin means ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.fill_between(t_flat, ci_lo, ci_hi, alpha=0.25, color="#2166ac",
                label="GP 95% credible interval")
ax.plot(t_flat, mu, color="#2166ac", linewidth=2, label="GP posterior mean")
ax.scatter(bins_train["midpoint"], bins_train["mean_xG"],
           s=bins_train["n"] / bins_train["n"].max() * 120,
           color="#d73027", zorder=5, label="Training bin means\n(size ∝ shots)")
ax.errorbar(bins_train["midpoint"], bins_train["mean_xG"],
            yerr=1.96 * np.sqrt(bins_train["se_sq"]),
            fmt="none", color="#d73027", linewidth=1, alpha=0.6)

# Mark the four 5-minute window boundaries
for boundary in [300, 600, 900]:
    ax.axvline(boundary, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)

ax.set_xlabel("Time into third period (seconds)", fontsize=11)
ax.set_ylabel("Mean xGoal per shot", fontsize=11)
ax.set_title("GP Regression: Within-Period Shot Quality Dynamics\n"
             "Trailing team, 2-goal deficit  |  Training: 2022–24", fontsize=11)
ax.set_xlim(0, PULL_CUTOFF)
ax.legend(fontsize=9, framealpha=0.9)

# Annotate window labels
for mid, lbl in zip([150, 450, 750, 1020], ["0–5 min", "5–10 min", "10–15 min", "15–18.5 min"]):
    ax.text(mid, ax.get_ylim()[0] + 0.001, lbl, ha="center", fontsize=7.5,
            color="grey", style="italic")

plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_gp_posterior.png", bbox_inches="tight")
plt.close()
print("Saved: FIG/fig6_gp_posterior.png")

# ── Fig 7: Exponential overlay ────────────────────────────────────────────────
if exp_fitted:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(t_flat, ci_lo, ci_hi, alpha=0.2, color="#2166ac",
                    label="GP 95% credible interval")
    ax.plot(t_flat, mu, color="#2166ac", linewidth=2, label="GP posterior mean")
    ax.plot(t_flat, exp_pred, color="#d73027", linewidth=2, linestyle="--",
            label=f"Exponential fit  (a={a_fit:.4f}, b={b_fit:.5f})")
    ax.scatter(bins_train["midpoint"], bins_train["mean_xG"],
               s=60, color="#555555", zorder=5, alpha=0.7, label="Training bin means")

    for boundary in [300, 600, 900]:
        ax.axvline(boundary, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)

    # Shade regions where exponential exits the CI
    outside_mask = ~inside
    ax.fill_between(t_flat, ci_lo, ci_hi,
                    where=outside_mask, alpha=0.35, color="#d73027",
                    label="Exponential outside CI")

    ax.set_xlabel("Time into third period (seconds)", fontsize=11)
    ax.set_ylabel("Mean xGoal per shot", fontsize=11)
    ax.set_title(f"Exponential Consistency Test\n"
                 f"{pct_inside:.1f}% of posterior inside CI  |  "
                 f"Training: 2022–24", fontsize=11)
    ax.set_xlim(0, PULL_CUTOFF)
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_exponential_overlay.png", bbox_inches="tight")
    plt.close()
    print("Saved: FIG/fig7_exponential_overlay.png")

# ── Fig 8: Validation posterior predictive check ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.fill_between(t_flat, ci_lo, ci_hi, alpha=0.2, color="#2166ac",
                label="Training GP 95% CI (2022–24)")
ax.plot(t_flat, mu, color="#2166ac", linewidth=2, label="Training GP mean")

# Validation bins: green = inside CI, red = outside
colors_v = ["#4dac26" if ins else "#d73027" for ins in inside_v]
ax.scatter(bins_valid["midpoint"], bins_valid["mean_xG"],
           c=colors_v, s=70, zorder=5,
           label=f"Validation bins (2024–25)  "
                 f"[{inside_v.sum()}/{len(inside_v)} inside CI]")
ax.errorbar(bins_valid["midpoint"], bins_valid["mean_xG"],
            yerr=1.96 * np.sqrt(bins_valid["se_sq"]),
            fmt="none", color="grey", linewidth=1, alpha=0.6)

for boundary in [300, 600, 900]:
    ax.axvline(boundary, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)

ax.set_xlabel("Time into third period (seconds)", fontsize=11)
ax.set_ylabel("Mean xGoal per shot", fontsize=11)
ax.set_title("Prospective Validation: 2024–25 Season\n"
             "vs. GP Posterior Predictive (trained on 2022–24)", fontsize=11)
ax.set_xlim(0, PULL_CUTOFF)
ax.legend(fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig8_validation_gp.png", bbox_inches="tight")
plt.close()
print("Saved: FIG/fig8_validation_gp.png")

# ── Fig 9: Semi-log plot ───────────────────────────────────────────────────────
# If the exponential model were correct, log(mean_xG) vs time would be linear.
# Plot log(bin mean) vs time for training data to visualise departure from linearity.

fig, ax = plt.subplots(figsize=(8, 5))

log_y     = np.log(bins_train["mean_xG"].values)
log_ci_lo = np.log(np.maximum(ci_lo, 1e-6))
log_ci_hi = np.log(ci_hi)

ax.fill_between(t_flat, log_ci_lo, log_ci_hi, alpha=0.2, color="#2166ac",
                label="GP 95% CI (log scale)")
ax.plot(t_flat, np.log(mu), color="#2166ac", linewidth=2, label="GP posterior mean")
ax.scatter(bins_train["midpoint"], log_y,
           s=60, color="#d73027", zorder=5, label="log(bin mean xGoal)")

# If exponential fit: log(y) = log(a) + b*t is a straight line — overlay it
if exp_fitted:
    ax.plot(t_flat, np.log(a_fit) + b_fit * t_flat,
            color="#d73027", linewidth=1.5, linestyle="--",
            label=f"Exponential (linear in log space)")

for boundary in [300, 600, 900]:
    ax.axvline(boundary, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)

ax.set_xlabel("Time into third period (seconds)", fontsize=11)
ax.set_ylabel("log(Mean xGoal per shot)", fontsize=11)
ax.set_title("Semi-Log Plot: Exponential Model Diagnostics\n"
             "(Linear trend = consistent with exponential growth)", fontsize=11)
ax.set_xlim(0, PULL_CUTOFF)
ax.legend(fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig9_semilog.png", bbox_inches="tight")
plt.close()
print("Saved: FIG/fig9_semilog.png")

print("\nDone. GP regression complete.")