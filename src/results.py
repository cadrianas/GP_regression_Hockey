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
    """Fit simple training curves and predict at the validation times.

    Validation supplies query midpoints only: its mean_xG values do not fit
    coefficients. Returned variances describe fitted-mean parameter uncertainty
    under the supplied sampling-error approximation, not new-observation noise.
    """
    # Express time in 10-minute units for the numerical baseline fits.
    x = train.midpoint.to_numpy() / 600
    xv = valid.midpoint.to_numpy() / 600
    y = train.mean_xG.to_numpy()
    # Treat estimated variances as fixed; floor them to avoid division by zero.
    variance = np.maximum(train.se_sq.to_numpy(), 1e-12)
    results = {}
    # These restricted shapes provide reference predictions for the flexible GP.
    # A constant cannot change with time; a line can change only at a fixed rate.
    # The design has an intercept only (constant) or intercept + time (linear).
    # Dividing by SE gives inverse-variance weighted least squares.
    for name, degree in [("Constant", 0), ("Linear", 1)]:
        design = np.vander(x, N=degree + 1, increasing=True)
        query = np.vander(xv, N=degree + 1, increasing=True)
        weighted_design = design / np.sqrt(variance[:, None])
        covariance = np.linalg.inv(weighted_design.T @ weighted_design)
        beta = np.linalg.lstsq(weighted_design, y / np.sqrt(variance), rcond=None)[0]
        results[name] = (query @ beta, np.einsum("ij,jk,ik->i", query, covariance, query))

    def exponential(t, a, b):
        return a * np.exp(b * t)

    # The exponential is a simple nonlinear alternative, fitted on train only.
    parameters, covariance = curve_fit(exponential, x, y, p0=[y.mean(), 0.0],
                                       sigma=np.sqrt(variance), absolute_sigma=True,
                                       bounds=([0, -10], [1, 10]), maxfev=10000)
    a, b = parameters
    # Propagate coefficient covariance to fitted-mean variance at query times.
    jacobian = np.column_stack([np.exp(b * xv), a * xv * np.exp(b * xv)])
    results["Exponential"] = (exponential(xv, a, b),
                               np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian))
    # Convert the exponential rate from per-10-minutes back to per-second.
    return results, b / 600


def analyze(full, output):
    """Compare fixed era-specific models with later-season observed bin means.

    All error metrics below weight evaluation bins equally. The later sample
    provides its observed means and sampling variances for retrospective checks;
    it does not fit the earlier curves. This is not a live game-state model.
    """
    # Build evaluation observations separately, before fitting either era.
    valid = make_bins(full[full.era == "validate"], "validate")
    rows = []
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    for era in ["early", "recent"]:
        # Only this era's training table is passed into the GP fit.
        train = make_bins(full[full.era == era], era)
        model = fit_gp(train)
        baselines, exponential_rate = baseline_predictions(train, valid)
        # Query times may come from validation; query outcomes are not inputs.
        mean, std = predict_latent(model, valid.midpoint)
        # For an OBSERVED held-out mean, add its estimated sampling variance.
        # The GP also includes fitted residual noise; baseline assumptions differ.
        predictions = {name: (mu, np.sqrt(np.maximum(var, 0) + valid.se_sq.to_numpy()))
                       for name, (mu, var) in baselines.items()}
        predictions["GP (Matérn 5/2)"] = (mean, predictive_std(model, std, valid.se_sq))
        fig, ax = plt.subplots(figsize=(9, 5))
        # Evaluate every model on the same bins. The final 30-second bin gets
        # the same metric weight as a minute bin, regardless of its shot count.
        for name, (mu, pred_std) in predictions.items():
            actual = valid.mean_xG.to_numpy()
            # Pointwise prediction intervals, not simultaneous curve bands.
            # Coverage and width are descriptive checks, not proof of calibration.
            lo, hi = mu - 1.96 * pred_std, mu + 1.96 * pred_std
            rows.append({"era": era, "model": name,
                         "rmse": np.sqrt(np.mean((actual - mu)**2)),
                         "mae": np.abs(actual - mu).mean(),
                         "prediction_coverage_pct": 100 * ((actual >= lo) & (actual <= hi)).mean(),
                         "mean_prediction_interval_width": (hi - lo).mean()})
            ax.plot(valid.midpoint, mu, label=name)
        # Black points are later-season observations, not the fitting targets.
        # Gaps from the lines describe prediction discrepancies; this plot alone
        # cannot distinguish underfitting from sampling variation or season shifts.
        # Assess underfitting with training fit/residuals as well as later errors.
        ax.scatter(valid.midpoint, valid.mean_xG, color="black", label="2024-25 observed bins")
        ax.set(xlabel="Seconds into third period", ylabel="Mean xGoal", title=f"Validation predictions: {era}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / f"model_comparison_{era}.png", dpi=150)
        plt.close(fig)

        # Optional shape diagnostics; these samples are conditional function
        # draws, not MCMC over hyperparameters or a formal exponential-shape test.
        # Median rate of change f'/f is a descriptive summary of the GP's implied shape
        # positive values indicate increasing scoring rate, 
        # negative values indicate decreasing scoring rate.
        grid = np.linspace(0, PULL_CUTOFF, 200)
        low, median, high = derivative_ratio_samples(model, grid)
        pd.DataFrame({"time_s": grid, "ratio_median": median, "ratio_lo": low,
                      "ratio_hi": high, "exponential_rate": exponential_rate}).to_csv(
                          output / f"derivative_ratio_{era}.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(grid, median, label="GP f'/f posterior median")
        ax.fill_between(grid, low, high, alpha=.2, label="Pointwise 95% posterior interval")
        ax.axhline(exponential_rate, linestyle="--", color="red", label="Fitted exponential rate (point estimate)")
        ax.set(xlabel="Seconds into third period", ylabel="f'/f (per second)",
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
