"""Describe the selected shots before interpreting any fitted GP curve.

Write per-season and pooled window summaries, exploratory season tests, and a
training-versus-later-season comparison. All selection rules come from analysis.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import (WINDOW_LABELS, WINDOW_MIDPOINTS, cluster_bootstrap,
                          decompose, run_inputs)


def compare_samples(train, valid, n_boot=2000):
    """Compare the same shot-weighted quantity, with uncertainty in both samples."""
    # The samples cover different seasons; use separate random redraw streams.
    # Each side retains whole-game dependence and the same shot-weighted target.
    tr, tr_draws = cluster_bootstrap(train, n_boot=n_boot, seed=42)
    va, va_draws = cluster_bootstrap(valid, n_boot=n_boot, seed=43)
    if not tr.index.equals(va.index):
        raise ValueError("Training and validation must populate the same windows")
    # Compare validation minus training directly, including sampling variation
    # on BOTH sides. Overlap of two separate intervals is not this contrast.
    delta = va_draws - tr_draws
    lo, hi = np.nanpercentile(delta, [2.5, 97.5], axis=0)
    return pd.DataFrame({
        "window": [WINDOW_LABELS[w] for w in tr.index],
        "train_mean_xG": tr["mean_xG"].to_numpy(),
        "train_ci_lo": tr["ci_lo"].to_numpy(), "train_ci_hi": tr["ci_hi"].to_numpy(),
        "valid_mean_xG": va["mean_xG"].to_numpy(),
        "valid_ci_lo": va["ci_lo"].to_numpy(), "valid_ci_hi": va["ci_hi"].to_numpy(),
        "difference": va["mean_xG"].to_numpy() - tr["mean_xG"].to_numpy(),
        "difference_ci_lo": lo, "difference_ci_hi": hi,
    })


def season_tests(train):
    """Exploratory tests of game-mean distributions, not independent shots.

    These test a different estimand from the shot-weighted summaries. Neither
    failure to reject nor rejection decides whether pooling is appropriate.
    """
    rows = []
    for w, label in enumerate(WINDOW_LABELS):
        # One mean per represented game, then group those means by season.
        # Games without an eligible attempt in this window do not enter the test.
        game_means = train[train.window == w].groupby(["season", "game_key"])["xGoal"].mean()
        samples = [part.to_numpy() for _, part in game_means.groupby(level=0)]
        f, pa, h, pk = np.nan, np.nan, np.nan, np.nan
        if len(samples) > 1 and all(len(s) > 1 for s in samples):
            if np.ptp(np.concatenate(samples)) > 0:
                # ANOVA addresses mean differences under its assumptions;
                # Kruskal-Wallis compares ranks, not specifically the means.
                # Neither identifies a causal change or a unique era boundary.
                f, pa = stats.f_oneway(*samples)
                h, pk = stats.kruskal(*samples)
        rows.append({"window": label, "f_stat": f, "p_anova": pa,
                     "h_stat": h, "p_kruskal": pk})
    result = pd.DataFrame(rows)
    # Conservative correction for inspecting four windows, separately per test.
    for col in ["p_anova", "p_kruskal"]:
        result[col + "_bonferroni"] = np.minimum(result[col] * len(rows), 1)
    return result


def analyze(full, output):
    """Return no model: this entry point saves descriptive tables and figures."""
    train = full[full.role == "train"]
    valid = full[full.role == "validate"]
    # Inspect each season as well as the pool; pooling can hide temporal variation.
    tables = pd.concat([decompose(df, label) for label, df in train.groupby("season")], ignore_index=True)
    pooled = decompose(train, "2014-15 through 2023-24")
    comparison = compare_samples(train, valid)
    tables.to_csv(output / "decomposition_by_season.csv", index=False)
    pooled.to_csv(output / "decomposition_pooled.csv", index=False)
    season_tests(train).to_csv(output / "pooling_stability_tests.csv", index=False)
    comparison.to_csv(output / "validation_check.csv", index=False)
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)

    # First plot: original units. Volume and accumulated xG are divided by
    # clock-window duration; mean xG is already a per-shot quantity.
    x = np.arange(4)
    columns = ["shots_per_game_per_window_minute", "mean_xG", "xG_per_game_per_window_minute"]
    labels = ["Shots / sampled game / window minute", "Mean xGoal per shot", "xG / sampled game / window minute"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, label in zip(axes, columns, labels):
        ax.bar(x, pooled[col])
        ax.set_xticks(x, WINDOW_LABELS, rotation=20)
        ax.set_ylabel(label)
    fig.suptitle("Comeback shots: clock-window summaries, not time-at-risk rates")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_decomposition_bars.png", dpi=150)
    plt.close(fig)

    # Indexed plot: each series is relative to its OWN first-window value.
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label in zip(columns, ["Window-normalized shot volume", "Shot quality", "Window-normalized xG"]):
        ax.plot(x, pooled["idx_" + col], "o-", label=label)
    ax.set_xticks(x, WINDOW_LABELS)
    ax.set_ylabel("Index relative to first window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig2_indexed_trajectories.png", dpi=150)
    plt.close(fig)

    # Heatmap exposes season differences that the pooled trajectory can conceal.
    pivot = tables.pivot(index="season", columns="window", values="mean_xG").reindex(columns=range(4))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(x, WINDOW_LABELS)
    ax.set_yticks(range(len(pivot)), pivot.index)
    fig.colorbar(im, ax=ax, label="Mean xGoal per shot")
    fig.tight_layout()
    fig.savefig(figdir / "fig_season_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    # These bands are percentile GAME-bootstrap intervals for broad-window means,
    # not GP latent intervals or a simultaneous confidence band across windows.
    for prefix, label in [("train", "Training"), ("valid", "Validation")]:
        ax.plot(x, comparison[prefix + "_mean_xG"], "o-", label=label)
        ax.fill_between(x, comparison[prefix + "_ci_lo"].to_numpy(),
                        comparison[prefix + "_ci_hi"].to_numpy(), alpha=.2)
    ax.set_xticks(x, WINDOW_LABELS)
    ax.set_ylabel("Mean xGoal per shot")
    ax.set_title("Pointwise 95% game-cluster bootstrap intervals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig5_validation.png", dpi=150)
    plt.close(fig)
    print(comparison.to_string(index=False))
    print("Volume is normalized by clock-window duration, not time spent trailing by two.")
    print(f"Saved analysis to {output}")


def main():
    # Shared CLI validates and loads all configured seasons before reporting.
    full, output = run_inputs(__doc__, "decomposition")
    analyze(full, output)


# Importing helpers must not start a full-data analysis or create output files.
if __name__ == "__main__":
    main()
