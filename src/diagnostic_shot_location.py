"""
Diagnostic: Shot Location & Goalie Pull Patterns
=================================================

Testing the hypothesis: Why did xGoal quality spike in 2022-23?

Key investigations:
  1. Goalie-pull frequency (direct detection from skater counts)
  2. Shot distance distribution (are shots closer to net?)
  3. Shot angle distribution (are shots from higher-danger areas?)
  4. Danger zone concentration (slot vs. perimeter)
  5. Shot type distribution (more tips, redirects, screen shots?)

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists():
    ROOT = ROOT.parent

sys.path.append(str(ROOT))

from src.paths import DATA_DIR, RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "diagnostics"
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────

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

PERIOD_DURATION = 1200
PULL_CUTOFF = 1110

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ── Load Data ─────────────────────────────────────────────────────────────────

def load_and_filter(filepath: str, season_label: str) -> pd.DataFrame:
    """Load MoneyPuck shots CSV and apply the comeback filter."""
    print(f"  Loading {filepath} ({season_label})...")
    df = pd.read_csv(filepath)

    df["period"] = df["period"].astype(int)
    df["homeTeamGoals"] = df["homeTeamGoals"].astype(int)
    df["awayTeamGoals"] = df["awayTeamGoals"].astype(int)

    mask = (
            (df["period"] == 3) &
            (abs(df["homeTeamGoals"] - df["awayTeamGoals"]) == 2)
    )
    df3 = df[mask].copy()

    df3["trailingTeam"] = np.where(
        df3["homeTeamGoals"] < df3["awayTeamGoals"],
        df3["homeTeamCode"],
        df3["awayTeamCode"],
    ).astype(str)
    df3 = df3[df3["teamCode"].astype(str) == df3["trailingTeam"]].copy()

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

    print(f"    → {len(df3):,} shots")
    return df3


print("Loading comeback situation data...\n")
all_dfs = []
for filepath, meta in SEASONS.items():
    if not os.path.exists(filepath):
        print(f"  SKIPPING {filepath}")
        continue
    df_season = load_and_filter(filepath, meta["label"])
    df_season["role"] = meta["role"]
    all_dfs.append(df_season)

full = pd.concat(all_dfs, ignore_index=True)
train = full[full["role"] == "train"].copy()

print(f"\nTotal training shots: {len(train):,}")

# ── 1. GOALIE PULL FREQUENCY (Direct Detection) ───────────────────────────────

print("\n" + "=" * 70)
print("1. GOALIE PULL FREQUENCY (via skater count)")
print("=" * 70 + "\n")


# Detect goalie pulls: if trailing team has fewer skaters, goalie was pulled
def detect_skater_ratio(row):
    """Detect if goalie pull occurred (trailing team has 6 skaters vs defense's 5)."""
    trailing = row["trailingTeam"]
    home_code = row["homeTeamCode"]

    # Get skater counts for trailing team
    if trailing == home_code:
        trailing_skaters = row["homeSkatersOnIce"]
        defending_skaters = row["awaySkatersOnIce"]
    else:
        trailing_skaters = row["awaySkatersOnIce"]
        defending_skaters = row["homeSkatersOnIce"]

    # Goalie pull: trailing team has 6, defending has 5 (or trailing has 5, defending has 4 in rare cases)
    is_pull = trailing_skaters > defending_skaters

    return pd.Series({
        "trailing_skaters": trailing_skaters,
        "defending_skaters": defending_skaters,
        "is_goalie_pull": is_pull,
    })


train[["trailing_skaters", "defending_skaters", "is_goalie_pull"]] = \
    train.apply(detect_skater_ratio, axis=1)

pull_by_season = train.groupby("season").apply(
    lambda df: (df["is_goalie_pull"].sum() / len(df)) * 100
)

print("Goalie-pull frequency (goalie was pulled):\n")
print("  Season     | Pull % | vs Early Era")
print("  " + "-" * 50)

early_avg = pull_by_season[pull_by_season.index.isin(["2014-15", "2015-16", "2016-17",
                                                      "2017-18", "2018-19", "2019-20"])].mean()

for season, pct in pull_by_season.items():
    diff = pct - early_avg
    arrow = "↑" if diff > 1 else "↓" if diff < -1 else "→"
    print(f"  {season}  |  {pct:5.2f}% | {diff:+5.2f}% {arrow}")

print(f"\nEarly era (2014-20) avg: {early_avg:.2f}%")
print(f"Recent era (2022-24) avg: {pull_by_season[['2022-23', '2023-24']].mean():.2f}%")

# Save results
pull_summary = pd.DataFrame({
    "season": pull_by_season.index,
    "goalie_pull_pct": pull_by_season.values,
})
pull_summary.to_csv(OUTPUT_DIR / "goalie_pull_frequency.csv", index=False)
print("\nSaved: goalie_pull_frequency.csv")

# ── 2. SHOT DISTANCE ANALYSIS ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("2. SHOT DISTANCE DISTRIBUTION")
print("=" * 70 + "\n")

print("Mean shot distance (feet) by season:\n")
print("  Season     | Mean | Median | Q25 | Q75 | % < 20 ft (danger zone)")
print("  " + "-" * 70)

distance_by_season = []
for season in sorted(train["season"].unique()):
    df_s = train[train["season"] == season]
    shot_dist = df_s["shotDistance"].dropna()

    if len(shot_dist) > 0:
        mean_dist = shot_dist.mean()
        median_dist = shot_dist.median()
        q25 = shot_dist.quantile(0.25)
        q75 = shot_dist.quantile(0.75)
        pct_close = (shot_dist < 20).sum() / len(shot_dist) * 100

        distance_by_season.append({
            "season": season,
            "mean_distance": mean_dist,
            "median_distance": median_dist,
            "q25": q25,
            "q75": q75,
            "pct_danger_zone": pct_close,
            "n_shots": len(shot_dist),
        })

        print(f"  {season}  | {mean_dist:5.2f} | {median_dist:6.2f} | {q25:4.1f} | {q75:4.1f} | {pct_close:5.1f}%")

distance_df = pd.DataFrame(distance_by_season)
distance_df.to_csv(OUTPUT_DIR / "shot_distance_by_season.csv", index=False)

# Test for significant shift in 2022-23
early_dist = train[train["season"].isin(["2014-15", "2015-16", "2016-17",
                                         "2017-18", "2018-19", "2019-20"])]["shotDistance"].dropna()
recent_dist = train[train["season"].isin(["2022-23", "2023-24"])]["shotDistance"].dropna()

t_stat, p_val = stats.ttest_ind(early_dist, recent_dist)
mw_stat, p_val_mw = stats.mannwhitneyu(early_dist, recent_dist)

print(f"\nTest: Early (2014-20) vs Recent (2022-24)")
print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
print(f"  Mann-Whitney U: U={mw_stat:.0f}, p={p_val_mw:.4f}")

if p_val_mw < 0.05:
    print(f"  ✓ SIGNIFICANT shift in shot distance distribution")
else:
    print(f"  ✗ NO significant shift in shot distance")

print("\nSaved: shot_distance_by_season.csv\n")

# ── 3. DANGER ZONE CONCENTRATION ───────────────────────────────────────────────

print("=" * 70)
print("3. DANGER ZONE CONCENTRATION")
print("=" * 70 + "\n")

# Define danger zones by distance
# Slot/Home plate: < 20 ft
# Perimeter: 20-30 ft
# Far: > 30 ft

danger_zones = []
for season in sorted(train["season"].unique()):
    df_s = train[train["season"] == season]
    shot_dist = df_s["shotDistance"].dropna()

    if len(shot_dist) > 0:
        slot = (shot_dist < 20).sum()
        perimeter = ((shot_dist >= 20) & (shot_dist < 30)).sum()
        far = (shot_dist >= 30).sum()

        total = len(shot_dist)

        danger_zones.append({
            "season": season,
            "slot_pct": (slot / total) * 100,
            "perimeter_pct": (perimeter / total) * 100,
            "far_pct": (far / total) * 100,
            "n_shots": total,
        })

        print(f"  {season}:")
        print(f"    Slot (< 20 ft):      {slot:4d} ({(slot / total) * 100:5.1f}%)")
        print(f"    Perimeter (20-30 ft): {perimeter:4d} ({(perimeter / total) * 100:5.1f}%)")
        print(f"    Far (> 30 ft):       {far:4d} ({(far / total) * 100:5.1f}%)")

danger_df = pd.DataFrame(danger_zones)
danger_df.to_csv(OUTPUT_DIR / "danger_zone_by_season.csv", index=False)

print("\nSaved: danger_zone_by_season.csv\n")

# ── 4. SHOT TYPE DISTRIBUTION ──────────────────────────────────────────────────

print("=" * 70)
print("4. SHOT TYPE DISTRIBUTION")
print("=" * 70 + "\n")

# Check most common shot types
print("Most common shot types across all periods:\n")
print(train["shotType"].value_counts().head(10))

print("\n\nShot type shifts in recent era (2022-24) vs early (2014-20):\n")

shot_types = train[train["shotType"].notna()]["shotType"].unique()
shot_types = sorted([st for st in shot_types if pd.notna(st)])[:10]  # Top 10

early_shots = train[train["season"].isin(["2014-15", "2015-16", "2016-17",
                                          "2017-18", "2018-19", "2019-20"])]
recent_shots = train[train["season"].isin(["2022-23", "2023-24"])]

print("  Shot Type              | Early % | Recent % | Δ")
print("  " + "-" * 55)

for stype in shot_types:
    early_pct = (early_shots["shotType"] == stype).sum() / len(early_shots) * 100
    recent_pct = (recent_shots["shotType"] == stype).sum() / len(recent_shots) * 100
    delta = recent_pct - early_pct
    arrow = "↑" if delta > 1 else "↓" if delta < -1 else "→"

    print(f"  {stype:20s} | {early_pct:6.2f}% | {recent_pct:7.2f}% | {delta:+5.2f}% {arrow}")

# ── 5. COORDINATE-BASED VISUALIZATION ──────────────────────────────────────────

print("\n" + "=" * 70)
print("5. SHOT LOCATION HEATMAPS")
print("=" * 70 + "\n")

# Create heatmaps of shot locations for early vs recent
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Shot Locations: Early Era vs Recent Era\n(Comeback situations, trailing team)",
             fontsize=12)

for ax, era_name, era_seasons in zip(
        axes,
        ["Early (2014-20)", "Recent (2022-24)"],
        [["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20"],
         ["2022-23", "2023-24"]],
):
    era_data = train[train["season"].isin(era_seasons)]

    # Use adjusted coordinates (xCordAdjusted, yCordAdjusted)
    x = era_data["xCordAdjusted"].dropna()
    y = era_data["yCordAdjusted"].dropna()

    if len(x) > 0 and len(y) > 0:
        # Create 2D histogram
        h = ax.hist2d(x, y, bins=30, cmap="YlOrRd", cmin=1)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-42, 42)
        ax.set_aspect("equal")
        ax.set_xlabel("X coordinate (adjusted)", fontsize=9)
        ax.set_ylabel("Y coordinate (adjusted)", fontsize=9)
        ax.set_title(f"{era_name}\n(n={len(x):,} shots)", fontsize=10)
        plt.colorbar(h[3], ax=ax, label="Shot count")

plt.tight_layout()
plt.savefig(FIG_DIR / "shot_location_heatmap.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: shot_location_heatmap.png")

# ── 6. WINDOW-LEVEL ANALYSIS (IS THE SHIFT CONCENTRATED?) ────────────────────

print("\nWindow-level shot distance analysis:\n")

for w in range(4):
    window_labels = ["0–5 min", "5–10 min", "10–15 min", "15–20 min"]

    early = train[(train["window"] == w) &
                  (train["season"].isin(["2014-15", "2015-16", "2016-17",
                                         "2017-18", "2018-19", "2019-20"]))]["shotDistance"].dropna()
    recent = train[(train["window"] == w) &
                   (train["season"].isin(["2022-23", "2023-24"]))]["shotDistance"].dropna()

    if len(early) > 0 and len(recent) > 0:
        t_stat, p_val = stats.ttest_ind(early, recent)
        early_mean = early.mean()
        recent_mean = recent.mean()
        delta_pct = ((recent_mean - early_mean) / early_mean) * 100

        sig = "✓" if p_val < 0.05 else "✗"
        print(f"  Window {w} ({window_labels[w]:10s}): "
              f"early={early_mean:.2f} ft, recent={recent_mean:.2f} ft, "
              f"Δ={delta_pct:+.1f}%, p={p_val:.4f} {sig}")

# ── 7. SUMMARY TABLE ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: Key Metrics Comparison")
print("=" * 70 + "\n")

summary = pd.DataFrame({
    "Season": distance_df["season"],
    "Mean_Distance_ft": distance_df["mean_distance"].round(2),
    "Danger_Zone_pct": danger_df["slot_pct"].round(1),
    "Goalie_Pull_pct": pull_summary.set_index("season").loc[distance_df["season"], "goalie_pull_pct"].values.round(1),
    "Mean_xGoal": [train[train["season"] == s]["xGoal"].mean() for s in distance_df["season"]],
})

summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("All diagnostic outputs saved to:")
print(f"  {OUTPUT_DIR}")
print("=" * 70)