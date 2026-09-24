"""Descriptive goalie, location, and shot-type comparisons using shared eras."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import run_inputs, cluster_bootstrap, WINDOW_LABELS


def goalie_summary(train):
    """Share of sampled shots with the SHOOTING team's goalie absent.

    This is not the share of games with a goalie pull, nor time spent pulled.
    """
    result = train.groupby("season").agg(n_shots=("is_goalie_pull", "size"),
                                          goalie_absent_shots=("is_goalie_pull", "sum"))
    result["goalie_pull_shot_pct"] = 100 * result.goalie_absent_shots / result.n_shots
    return result.reset_index()


def analyze(full, output):
    train = full[full.role == "train"].copy()
    required = {"shotDistance", "shotType", "xCordAdjusted", "yCordAdjusted"}
    if not required.issubset(train.columns):
        raise ValueError(f"Missing diagnostic columns: {sorted(required - set(train.columns))}")
    goalie_summary(train).to_csv(output / "goalie_pull_frequency.csv", index=False)
    rows = []
    for season, df in train.groupby("season"):
        distance = df.shotDistance.dropna()
        rows.append({"season": season, "n_shots": len(df), "n_distance": len(distance),
                     "mean_distance": distance.mean(), "median_distance": distance.median(),
                     "close_shot_pct": 100 * (distance < 20).mean(),
                     "midrange_shot_pct": 100 * distance.between(20, 30, inclusive="left").mean(),
                     "far_shot_pct": 100 * (distance >= 30).mean(),
                     "mean_xGoal": df.xGoal.mean()})
    summary = pd.DataFrame(rows).merge(goalie_summary(train).drop(columns="n_shots"), on="season")
    summary.to_csv(output / "summary_metrics.csv", index=False)
    # Missing types are retained as a category, making the denominator explicit.
    train["shotType"] = train.shotType.fillna("Unknown")
    types = pd.crosstab(train.era, train.shotType, normalize="index") * 100
    types.to_csv(output / "shot_type_percentages.csv")

    # Compare shot-weighted distance with whole-game resampling, just as for xGoal.
    distance_data = train.dropna(subset=["shotDistance"]).copy()
    distance_data["xGoal"] = distance_data.shotDistance
    early, draws_e = cluster_bootstrap(distance_data[distance_data.era == "early"], seed=42)
    recent, draws_r = cluster_bootstrap(distance_data[distance_data.era == "recent"], seed=43)
    if not early.index.equals(recent.index):
        raise ValueError("Both eras must populate the same distance windows")
    lo, hi = np.nanpercentile(draws_r - draws_e, [2.5, 97.5], axis=0)
    pd.DataFrame({"window": [WINDOW_LABELS[w] for w in early.index],
                  "early_distance_ft": early.mean_xG.to_numpy(),
                  "recent_distance_ft": recent.mean_xG.to_numpy(),
                  "difference_lo": lo, "difference_hi": hi}).to_csv(
                      output / "distance_comparison.csv", index=False)

    # Shared spatial bins and color scale; drop coordinate pairs together.
    edges = [np.linspace(-100, 100, 41), np.linspace(-42.5, 42.5, 35)]
    histograms = {}
    for era in ["early", "recent"]:
        xy = train.loc[train.era == era, ["xCordAdjusted", "yCordAdjusted"]].dropna()
        if xy.empty:
            raise ValueError(f"No paired shot coordinates for {era}")
        hist, _, _ = np.histogram2d(xy.iloc[:, 0], xy.iloc[:, 1], bins=edges)
        if hist.sum() == 0:
            raise ValueError(f"No coordinates within rink bounds for {era}")
        histograms[era] = 100 * hist / hist.sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    vmax = max(h.max() for h in histograms.values())
    for ax, (era, hist) in zip(axes, histograms.items()):
        im = ax.pcolormesh(*edges, hist.T, cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set(title=era.title(), xlabel="Adjusted X (ft)", ylabel="Adjusted Y (ft)", aspect="equal")
    fig.colorbar(im, ax=axes, label="Percent of era's shots with coordinates in rink bounds")
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "shot_location_heatmap.png", dpi=150)
    plt.close(fig)
    print(summary.to_string(index=False))
    print("Distance <20 ft is a proximity category, not a geometric definition of the slot.")
    print("Goalie-pull percentages describe sampled shots through 18:30, not all goalie pulls.")


def main():
    full, output = run_inputs(__doc__, "diagnostics")
    analyze(full, output)


if __name__ == "__main__":
    main()
