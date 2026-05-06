"""
Decomposition Analysis: Volume vs. Quality in NHL Comeback Situations
Multi-Season Version with Prospective Validation
=====================================================================


MoneyPuck file naming convention: shots_{ENDING_YEAR}.csv
e.g. shots_2024.csv = the 2023-24 season

Outputs
-------
  decomposition_by_season.csv   Per-season decomposition tables stacked
  decomposition_pooled.csv      Training seasons pooled
  validation_check.csv          Held-out season vs. training bootstrap CI
  fig1_decomposition_bars.png   Side-by-side bars (pooled training)
  fig2_indexed_trajectories.png Volume / quality / total xG indexed
  fig3_xgoal_histogram.png      xGoal distribution by season
  fig4_season_consistency.png   Per-season mean xG per window (sanity check)
  fig5_validation.png           2025-26 vs. training 95% bootstrap CI

Dependencies: pandas, numpy, matplotlib, scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import sys

# Find project root dynamically
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent

sys.path.append(str(ROOT))



from src.paths import DATA_DIR, RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "decomposition"
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
# ── 0. Configuration ──────────────────────────────────────────────────────────

# Map file path → season label and role
# MoneyPuck naming: shots_{YEAR}.csv = season ending in {YEAR}

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

PERIOD_DURATION  = 1200
WINDOW_SIZE      = 300        # 5-minute bins
N_WINDOWS        = 4
WINDOW_LABELS    = ["0–5 min", "5–10 min", "10–15 min", "15–20 min"]
WINDOW_MIDPOINTS = [150, 450, 750, 1050]   # seconds; fed to GP regression later

# Exclude final ~90 s to avoid goalie-pull regime contamination
PULL_CUTOFF = 1110


plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ── 1. Load & Filter All Seasons ──────────────────────────────────────────────

def load_and_filter(filepath: str, season_label: str) -> pd.DataFrame:
    """Load one MoneyPuck shots CSV and apply the comeback filter."""
    print(f"  Loading {filepath} ({season_label})...")
    df = pd.read_csv(filepath)

    # Cast to int — MoneyPuck sometimes stores these as floats (3.0 etc.)
    df["period"]        = df["period"].astype(int)
    df["homeTeamGoals"] = df["homeTeamGoals"].astype(int)
    df["awayTeamGoals"] = df["awayTeamGoals"].astype(int)

    mask = (
        (df["period"] == 3) &
        (abs(df["homeTeamGoals"] - df["awayTeamGoals"]) == 2)
    )
    df3 = df[mask].copy()

    # Identify and keep only the trailing team's shots
    # Cast to str explicitly — np.where returns object array which can fail
    # equality checks against pandas StringArray dtype
    df3["trailingTeam"] = np.where(
        df3["homeTeamGoals"] < df3["awayTeamGoals"],
        df3["homeTeamCode"],
        df3["awayTeamCode"],
    ).astype(str)
    df3 = df3[df3["teamCode"].astype(str) == df3["trailingTeam"]].copy()

    # time is cumulative game seconds; period 3 starts at 2400 s.
    # Convert to time-within-period for windowing and the pull cutoff.
    df3["time_in_period"] = df3["time"] - 2400

    df3 = df3[df3["time_in_period"] <= PULL_CUTOFF].copy()

    df3["season"] = season_label

    df3["window"] = pd.cut(
        df3["time_in_period"],
        bins=[0, 300, 600, 900, 1200],
        labels=[0, 1, 2, 3],
        right=True,
        include_lowest=True,
    )

    print(f"    → {len(df3):,} shots, {df3['game_id'].nunique():,} games")
    return df3


print("Loading data...")
all_dfs = []
for filepath, meta in SEASONS.items():
    if not os.path.exists(filepath):
        print(f"  SKIPPING {filepath} — file not found. "
              f"Download from moneypuck.com/data.htm to include {meta['label']}.")
        continue
    df_season = load_and_filter(filepath, meta["label"])
    df_season["role"] = meta["role"]
    all_dfs.append(df_season)

full  = pd.concat(all_dfs, ignore_index=True)
train = full[full["role"] == "train"].copy()
valid = full[full["role"] == "validate"].copy()

print(f"\nTraining set:   {len(train):,} shots, {train['game_id'].nunique():,} games")
print(f"Validation set: {len(valid):,} shots, {valid['game_id'].nunique():,} games")

# ── 2. Decomposition Function ─────────────────────────────────────────────────

def decompose(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Four-window decomposition: total xG = shots × mean xGoal per shot.
    Returns one row per window with indexed metrics (window 0 = 1.00).
    """
    n_games = df["game_id"].nunique()

    agg = (
        df.groupby("window", observed=True)["xGoal"]
        .agg(shots="count", total_xG="sum", mean_xG="mean",
             median_xG="median", std_xG="std")
        .reset_index()
    )

    agg["shots_per_game"] = agg["shots"]    / n_games
    agg["xG_per_game"]    = agg["total_xG"] / n_games
    agg["n_games"]        = n_games
    agg["season"]         = label
    agg["window_label"]   = WINDOW_LABELS
    agg["midpoint_s"]     = WINDOW_MIDPOINTS

    # Index each metric to window 0 = 1.00 for relative comparisons
    for col in ["shots_per_game", "mean_xG", "xG_per_game"]:
        base = agg.loc[agg["window"] == 0, col].values[0]
        agg[f"idx_{col}"] = agg[col] / base

    return agg

# ── 3. Per-Season Decompositions (Training Sanity Check) ──────────────────────
# Run each training season separately first. If the within-period pattern
# differs substantially between 2023-24 and 2024-25, pooling is not justified
# and you would need a hierarchical model instead.

print("\n=== Per-Season Decomposition (Training) ===")
season_tables = []
for season_label in train["season"].unique():
    df_s  = train[train["season"] == season_label]
    tbl   = decompose(df_s, season_label)
    season_tables.append(tbl)
    print(f"\n  {season_label}")
    print(tbl[["window_label", "shots_per_game", "mean_xG", "xG_per_game"]]
          .to_string(index=False, float_format="{:.4f}".format))

season_summary = pd.concat(season_tables, ignore_index=True)
season_summary.to_csv("decomposition_by_season.csv", index=False, float_format="%.4f")
print("\nSaved: decomposition_by_season.csv")

# ── 4. Pooled Training Decomposition ─────────────────────────────────────────

print("\n=== Pooled Training Decomposition (2023-25) ===")
pooled = decompose(train, "2023-25 (pooled)")
print(pooled[["window_label", "shots_per_game", "mean_xG", "xG_per_game",
              "idx_shots_per_game", "idx_mean_xG", "idx_xG_per_game"]]
      .to_string(index=False, float_format="{:.4f}".format))
pooled.to_csv("decomposition_pooled.csv", index=False, float_format="%.4f")
print("Saved: decomposition_pooled.csv")

# ── 5. Bootstrap Confidence Intervals (Training) ──────────────────────────────
# We bootstrap at the GAME level, not the shot level, to respect within-game
# shot correlation. Shots from the same game are not independent — using
# shot-level bootstrap would understate uncertainty.
#
# We use percentile bootstrap (not t-bootstrap) because xGoal distributions
# are right-skewed and normality cannot be assumed.

N_BOOT = 2000
rng    = np.random.default_rng(seed=42)

print("\nBootstrapping game-level CIs for training windows...")

train_ci = {}   # window_index → (lo_95, hi_95, mean)

for w in range(N_WINDOWS):
    window_shots = train[train["window"] == w][["game_id", "xGoal"]]
    # One mean xGoal per game per window — this is the unit of resampling
    game_means = window_shots.groupby("game_id")["xGoal"].mean()
    n_g        = len(game_means)

    boot_means = np.array([
        game_means.sample(n=n_g, replace=True, random_state=rng.integers(1e9)).mean()
        for _ in range(N_BOOT)
    ])

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    train_ci[w] = (lo, hi, game_means.mean())
    print(f"  {WINDOW_LABELS[w]:12s}  mean={game_means.mean():.4f}  "
          f"95% CI [{lo:.4f}, {hi:.4f}]  (n_games={n_g})")

# ── 6. Prospective Validation ─────────────────────────────────────────────────
# For each window, ask: does the 2025-26 observed mean xGoal fall within
# the training bootstrap CI?
# A window outside the CI is a finding, not a failure — it means the
# within-period dynamic shifted in the current season.

print("\n=== Prospective Validation: 2025-26 vs. Training CI ===")
valid_decomp = decompose(valid, "2025-26 (validation)")

val_rows = []
for w in range(N_WINDOWS):
    val_mean        = valid_decomp.loc[valid_decomp["window"] == w, "mean_xG"].values[0]
    lo, hi, tr_mean = train_ci[w]
    inside          = lo <= val_mean <= hi
    val_rows.append({
        "window":        WINDOW_LABELS[w],
        "train_mean_xG": tr_mean,
        "train_ci_lo":   lo,
        "train_ci_hi":   hi,
        "valid_mean_xG": val_mean,
        "inside_95CI":   inside,
    })
    flag = "✓" if inside else "✗  ← outside CI"
    print(f"  {WINDOW_LABELS[w]:12s}  train={tr_mean:.4f} [{lo:.4f},{hi:.4f}]  "
          f"valid={val_mean:.4f}  {flag}")

val_check = pd.DataFrame(val_rows)
val_check.to_csv("validation_check.csv", index=False, float_format="%.4f")
print("Saved: validation_check.csv")

# ── 7. Interpretation Helper ──────────────────────────────────────────────────

delta_vol  = (pooled.loc[3, "idx_shots_per_game"] - 1) * 100
delta_qual = (pooled.loc[3, "idx_mean_xG"]        - 1) * 100
delta_tot  = (pooled.loc[3, "idx_xG_per_game"]    - 1) * 100
dominant   = "VOLUME" if abs(delta_vol) > abs(delta_qual) else "QUALITY"

print(f"\n=== Interpretation (Pooled Training) ===")
print(f"  Window 1 → Window 4:")
print(f"    Volume  change: {delta_vol:+.1f}%")
print(f"    Quality change: {delta_qual:+.1f}%")
print(f"    Total xG:       {delta_tot:+.1f}%")
print(f"  Dominant driver: {dominant}")

# ── 8. Figures ────────────────────────────────────────────────────────────────

x = np.arange(N_WINDOWS)

# Fig 1: Pooled training — three-panel decomposition bars
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(
    "Decomposition of Expected Goals — Comeback Situations\n"
    "Training: 2023–24 + 2024–25 (pooled)",
    fontsize=11, y=1.02,
)
for ax, col, ylabel, title in zip(
    axes,
    ["shots_per_game", "mean_xG", "xG_per_game"],
    ["Shots per game", "Mean xGoal per shot", "Total xGoals per game"],
    ["Shot Volume", "Shot Quality", "Volume × Quality"],
):
    bars = ax.bar(x, pooled[col], color="#2166ac", alpha=0.75, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(WINDOW_LABELS, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for bar, val in zip(bars, pooled[col]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + pooled[col].max()*0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_decomposition_bars.png", bbox_inches="tight")
plt.close()
print("Saved: fig1_decomposition_bars.png")

# Fig 2: Indexed trajectories (pooled training)
fig, ax = plt.subplots(figsize=(7, 4))
for col, label, fmt, color in [
    ("idx_shots_per_game", "Shot Volume",  "o-",  "#2166ac"),
    ("idx_mean_xG",        "Shot Quality", "s--", "#d73027"),
    ("idx_xG_per_game",    "Total xG",     "^:",  "#4dac26"),
]:
    ax.plot(x, pooled[col], fmt, color=color, linewidth=2, markersize=7, label=label)
ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(WINDOW_LABELS)
ax.set_ylabel("Index  (Window 1 = 1.00)", fontsize=10)
ax.set_xlabel("Time window in third period", fontsize=10)
ax.set_title("Relative Change: Volume, Quality, Total xG\n(Pooled training: 2023–25)", fontsize=10)
ax.legend(framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_indexed_trajectories.png", bbox_inches="tight")
plt.close()
print("Saved: fig2_indexed_trajectories.png")

# Fig 3: xGoal histograms by season
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
fig.suptitle("xGoal Distribution — Trailing-Team Shots (Period 3)", fontsize=11)
for ax, (_, meta) in zip(axes, SEASONS.items()):
    sl   = meta["label"]
    data = full[full["season"] == sl]["xGoal"]
    ax.hist(data, bins=50, color="#2166ac", alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.axvline(data.mean(),   color="#d73027", linewidth=1.5, linestyle="--",
               label=f"Mean={data.mean():.3f}")
    ax.axvline(data.median(), color="#4dac26", linewidth=1.5, linestyle=":",
               label=f"Median={data.median():.3f}")
    role_tag = " [VAL]" if meta["role"] == "validate" else ""
    ax.set_title(f"{sl}{role_tag}\n(N={len(data):,}, skew={data.skew():.2f})", fontsize=9)
    ax.set_xlabel("xGoal per shot", fontsize=8)
    ax.set_ylabel("Shots", fontsize=8)
    ax.legend(fontsize=7.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_xgoal_histogram.png", bbox_inches="tight")
plt.close()
print("Saved: fig3_xgoal_histogram.png")

# Fig 4: Per-season consistency (sanity check for pooling assumption)
fig, ax = plt.subplots(figsize=(7, 4))
palette   = ["#2166ac", "#d73027", "#4dac26", "#984ea3"]
colors_s  = {s_tbl["season"].iloc[0]: palette[i] for i, s_tbl in enumerate(season_tables)}
for s_tbl in season_tables:
    sl = s_tbl["season"].iloc[0]
    ax.plot(x, s_tbl["mean_xG"], "o-", color=colors_s[sl],
            linewidth=2, markersize=7, label=sl)
ax.set_xticks(x); ax.set_xticklabels(WINDOW_LABELS)
ax.set_ylabel("Mean xGoal per shot", fontsize=10)
ax.set_xlabel("Time window in third period", fontsize=10)
ax.set_title("Season-Level Consistency Check\n"
             "(If curves diverge substantially, pooling is not justified)", fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_season_consistency.png", bbox_inches="tight")
plt.close()
print("Saved: fig4_season_consistency.png")

# Fig 5: Prospective validation — 2025-26 vs. training 95% CI
fig, ax = plt.subplots(figsize=(7, 4))
tr_means = [train_ci[w][2] for w in range(N_WINDOWS)]
ci_lo    = [train_ci[w][0] for w in range(N_WINDOWS)]
ci_hi    = [train_ci[w][1] for w in range(N_WINDOWS)]
vl_means = val_check["valid_mean_xG"].values

ax.fill_between(x, ci_lo, ci_hi, alpha=0.25, color="#2166ac",
                label="Training 95% bootstrap CI")
ax.plot(x, tr_means, "o-", color="#2166ac", linewidth=2,
        markersize=7, label="Training mean (2023–25)")
ax.plot(x, vl_means, "s--", color="#d73027", linewidth=2,
        markersize=7, label="Validation (2025–26, partial)")

ax.set_xticks(x); ax.set_xticklabels(WINDOW_LABELS)
ax.set_ylabel("Mean xGoal per shot", fontsize=10)
ax.set_xlabel("Time window in third period", fontsize=10)
ax.set_title("Prospective Validation: 2025–26 Season\n"
             "vs. Training Bootstrap CI (2023–25)", fontsize=10)
ax.legend(framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_validation.png", bbox_inches="tight")
plt.close()
print("Saved: fig5_validation.png")

print("\nDone. All outputs saved to current directory.")
