import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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


train["is_empty_net"] = train["awayTeamGoals"] > train["homeTeamGoals"]  # or vice versa
# (This is a proxy; MoneyPuck may have an explicit flag)

empty_net_by_season = train.groupby("season").apply(
    lambda df: (df["is_empty_net"].sum() / len(df)) * 100
)

print("\nEmpty net shots as % of all trailing-team shots:\n")
for season, pct in empty_net_by_season.items():
    change = (pct - empty_net_by_season.iloc[0]) if pct != empty_net_by_season.iloc[0] else 0
    flag = "↑" if change > 1 else ""
    print(f"  {season}: {pct:.2f}%  {flag}")