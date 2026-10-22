"""Check GP residual-noise bounds while holding data, kernel form, and seed fixed."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import (PULL_CUTOFF, fit_gp, make_bins, predict_latent,
                          predictive_std, run_inputs)


def analyze(full, output):
    """Refit the same data under four noise bounds; save a stability summary.

    This checks one numerical modelling choice. It does not choose a bound by
    validation performance or establish robustness to other kernels or samples.
    """
    valid = make_bins(full[full.era == "validate"], "validate")
    grid = np.linspace(0, PULL_CUTOFF, 500)
    rows = []
    for era in ["early", "recent"]:
        bins = make_bins(full[full.era == era], era)
        # The default fit is the reference for pointwise curve changes.
        reference = fit_gp(bins)
        ref_mean, ref_std = predict_latent(reference, grid)
        # Keep the data, smoothness family, seed, and upper noise bound fixed.
        # Bounds are variances on the GP's NORMALISED target scale.
        for lower in [1e-10, 1e-8, 1e-6, 1e-4]:
            model = reference if lower == 1e-6 else fit_gp(bins, noise_bounds=(lower, 1.0))
            mean, std = predict_latent(model, grid)
            # Report later-season errors and coverage as diagnostics, not a
            # tuning rule for selecting whichever bound makes them look best.
            vm, vs = predict_latent(model, valid.midpoint)
            ps = predictive_std(model, vs, valid.se_sq)
            rows.append({
                "era": era, "noise_lower_bound": lower,
                "fitted_noise_normalized": model.kernel_.k2.noise_level,
                "max_mean_shift_vs_default": float(abs(mean - ref_mean).max()),
                "max_latent_std_shift_vs_default": float(abs(std - ref_std).max()),
                "validation_rmse": float(np.sqrt(np.mean((valid.mean_xG - vm)**2))),
                "bins_inside_prediction_interval": int((abs(valid.mean_xG - vm) <= 1.96 * ps).sum()),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "noise_bound_sensitivity.csv", index=False)
    print(table.to_string(index=False))
    print("A targeted noise-bound check, not a full sensitivity analysis or a tuning rule.")
    return table


def main():
    full, output = run_inputs(__doc__, "")
    analyze(full, output)


if __name__ == "__main__":
    main()
