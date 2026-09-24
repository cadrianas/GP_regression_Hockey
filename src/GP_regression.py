"""Separate GP trajectories for 2014–21 and 2022–23 starting seasons."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import (PULL_CUTOFF, make_bins, fit_gp, predict_latent,
                          predictive_std, run_inputs)


def analyze(full, output):
    bins = {era: make_bins(full[full.era == era], era) for era in ["early", "recent", "validate"]}
    models = {era: fit_gp(bins[era], era) for era in ["early", "recent"]}
    for era, table in bins.items():
        table.to_csv(output / f"gp_bin_data_{era}.csv", index=False)
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    grid = np.linspace(0, PULL_CUTOFF, 500)
    curves, validations = {}, []
    titles = {"early": "2014–15 through 2021–22", "recent": "2022–23 through 2023–24"}
    for era, model in models.items():
        mean, std = predict_latent(model, grid)
        curves[era] = mean, std
        pd.DataFrame({"time_s": grid, "mu": mean, "std": std,
                      "ci_lo": mean - 1.96 * std, "ci_hi": mean + 1.96 * std}).to_csv(
                          output / f"gp_results_{era}.csv", index=False)
        valid = bins["validate"]
        vm, vs = predict_latent(model, valid.midpoint)
        ps = predictive_std(model, vs, valid.se_sq)
        validation = pd.DataFrame({"era": era, "midpoint": valid.midpoint,
                                   "observed": valid.mean_xG, "predicted": vm,
                                   "prediction_lo": vm - 1.96 * ps, "prediction_hi": vm + 1.96 * ps})
        validation["inside_prediction_interval"] = valid.mean_xG.between(
            validation.prediction_lo, validation.prediction_hi)
        validations.append(validation)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(grid, mean, label="GP latent mean")
        ax.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std, alpha=.2,
                        label="Pointwise 95% latent interval")
        ax.errorbar(bins[era].midpoint, bins[era].mean_xG,
                    yerr=1.96 * np.sqrt(bins[era].se_sq), fmt="o", label="Training bins ± 1.96 bootstrap SE")
        ax.errorbar(valid.midpoint, valid.mean_xG, yerr=1.96 * np.sqrt(valid.se_sq),
                    fmt="^", label="2024–25 bins ± 1.96 bootstrap SE")
        ax.set(xlabel="Seconds into third period", ylabel="Mean xGoal per shot",
               title=f"GP fit: {titles[era]}", xlim=(0, PULL_CUTOFF))
        ax.legend(fontsize=8)
        fig.tight_layout()
        name = "fig6a_gp_early.png" if era == "early" else "fig6b_gp_recent.png"
        fig.savefig(figdir / name, dpi=150)
        plt.close(fig)
        print(f"{era}: {validation.inside_prediction_interval.sum()}/{len(valid)} held-out bins inside prediction intervals")
    pd.concat(validations, ignore_index=True).to_csv(output / "gp_validation.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for era, (mean, std) in curves.items():
        ax.plot(grid, mean, label=titles[era])
        ax.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std, alpha=.2)
    ax.set(xlabel="Seconds into third period", ylabel="Mean xGoal per shot",
           title="Era comparison: pointwise 95% latent intervals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig7_gp_comparison.png", dpi=150)
    plt.close(fig)
    print("Interval coverage is descriptive; it does not establish tactical adaptation.")


def main():
    full, output = run_inputs(__doc__, "")
    analyze(full, output)


if __name__ == "__main__":
    main()
