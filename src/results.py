"""Compare era-specific models and summarize GP derivative-ratio uncertainty.

Reads the configured shot files directly, so comparisons cannot accidentally
combine old pooled outputs with the current era-specific models.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import (run_inputs, make_bins, fit_gp, predict_latent,
                          predictive_std, derivative_ratio_samples, PULL_CUTOFF)


def baseline_predictions(train, valid):
    """Mean predictions and parameter variances under known bin sampling errors."""
    x = train.midpoint.to_numpy() / 600
    xv = valid.midpoint.to_numpy() / 600
    y = train.mean_xG.to_numpy()
    variance = np.maximum(train.se_sq.to_numpy(), 1e-12)
    results = {}
    for name, degree in [("Constant", 0), ("Linear", 1)]:
        design = np.vander(x, N=degree + 1, increasing=True)
        query = np.vander(xv, N=degree + 1, increasing=True)
        weighted_design = design / np.sqrt(variance[:, None])
        covariance = np.linalg.inv(weighted_design.T @ weighted_design)
        beta = np.linalg.lstsq(weighted_design, y / np.sqrt(variance), rcond=None)[0]
        results[name] = (query @ beta, np.einsum("ij,jk,ik->i", query, covariance, query))

    def exponential(t, a, b):
        return a * np.exp(b * t)

    parameters, covariance = curve_fit(exponential, x, y, p0=[y.mean(), 0.0],
                                       sigma=np.sqrt(variance), absolute_sigma=True,
                                       bounds=([0, -10], [1, 10]), maxfev=10000)
    a, b = parameters
    jacobian = np.column_stack([np.exp(b * xv), a * xv * np.exp(b * xv)])
    results["Exponential"] = (exponential(xv, a, b),
                               np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian))
    return results, b / 600


def analyze(full, output):
    valid = make_bins(full[full.era == "validate"], "validate")
    rows = []
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    for era in ["early", "recent"]:
        train = make_bins(full[full.era == era], era)
        model = fit_gp(train)
        baselines, exponential_rate = baseline_predictions(train, valid)
        mean, std = predict_latent(model, valid.midpoint)
        predictions = {name: (mu, np.sqrt(np.maximum(var, 0) + valid.se_sq.to_numpy()))
                       for name, (mu, var) in baselines.items()}
        predictions["GP (Matérn 5/2)"] = (mean, predictive_std(model, std, valid.se_sq))
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, (mu, pred_std) in predictions.items():
            actual = valid.mean_xG.to_numpy()
            lo, hi = mu - 1.96 * pred_std, mu + 1.96 * pred_std
            rows.append({"era": era, "model": name,
                         "rmse": np.sqrt(np.mean((actual - mu)**2)),
                         "mae": np.abs(actual - mu).mean(),
                         "prediction_coverage_pct": 100 * ((actual >= lo) & (actual <= hi)).mean(),
                         "mean_prediction_interval_width": (hi - lo).mean()})
            ax.plot(valid.midpoint, mu, label=name)
        ax.scatter(valid.midpoint, valid.mean_xG, color="black", label="2024–25 observed bins")
        ax.set(xlabel="Seconds into third period", ylabel="Mean xGoal", title=f"Validation predictions: {era}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / f"model_comparison_{era}.png", dpi=150)
        plt.close(fig)

        grid = np.linspace(0, PULL_CUTOFF, 200)
        low, median, high = derivative_ratio_samples(model, grid)
        pd.DataFrame({"time_s": grid, "ratio_median": median, "ratio_lo": low,
                      "ratio_hi": high, "exponential_rate": exponential_rate}).to_csv(
                          output / f"derivative_ratio_{era}.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(grid, median, label="GP f′/f posterior median")
        ax.fill_between(grid, low, high, alpha=.2, label="Pointwise 95% posterior interval")
        ax.axhline(exponential_rate, linestyle="--", color="red", label="Fitted exponential rate (point estimate)")
        ax.set(xlabel="Seconds into third period", ylabel="f′/f (per second)",
               title=f"Descriptive shape comparison: {era}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / f"derivative_ratio_{era}.png", dpi=150)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "model_comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Prediction intervals include held-out sampling variance. Baselines assume their mean forms are correct;")
    print("the GP additionally estimates residual white noise. Shape intervals are pointwise, not a formal rejection test.")


def main():
    full, output = run_inputs(__doc__, "model_comparison")
    analyze(full, output)


if __name__ == "__main__":
    main()
