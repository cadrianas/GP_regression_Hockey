"""Generate job-talk sensitivity checks, GP hyperparameter tables, and slide-ready figures.

Runs:
1. Manpower stratification (all situations vs. goalie-in-net vs. 5v5).
2. Era-boundary sensitivity across alternative recent-era start years.
3. Extraction of fitted Matérn 5/2 kernel hyperparameters.
4. Four clean, presentation-ready figures under <output-dir>/figures/.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import (PULL_CUTOFF, WINDOW_LABELS, decompose, fit_gp,
                          gp_hyperparameters, make_bins, predict_latent,
                          predictive_std, run_inputs)
from src.results import baseline_predictions


def _style_axis(ax):
    """Apply clean, presentation-friendly axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


def _season_start_year(season_label):
    return int(str(season_label).split("-")[0])


def evaluate_stratum(train_df, valid_df, era_label, stratum_label):
    """Fit a GP on one stratum and evaluate on the matching held-out stratum."""
    train_bins = make_bins(train_df, f"{era_label}_{stratum_label}")
    valid_bins = make_bins(valid_df, f"validate_{stratum_label}")
    gp = fit_gp(train_bins, f"{era_label}_{stratum_label}")
    mean, std = predict_latent(gp, valid_bins.midpoint)
    pred_std = predictive_std(gp, std, valid_bins.se_sq)
    actual = valid_bins.mean_xG.to_numpy()
    lo, hi = mean - 1.96 * pred_std, mean + 1.96 * pred_std

    decomp = decompose(train_df, f"{era_label}_{stratum_label}")
    w0 = float(decomp.loc[0, "mean_xG"])
    w3 = float(decomp.loc[3, "mean_xG"])
    pct_jump = 100.0 * (w3 - w0) / w0 if w0 > 0 else np.nan

    params = gp_hyperparameters(gp)
    summary = {
        "era": era_label,
        "stratum": stratum_label,
        "n_shots": int(len(train_df)),
        "n_games": int(train_df["game_key"].nunique()),
        "w0_mean_xG": w0,
        "w3_mean_xG": w3,
        "pct_increase_w0_to_w3": pct_jump,
        "val_rmse": float(np.sqrt(np.mean((actual - mean) ** 2))),
        "val_mae": float(np.abs(actual - mean).mean()),
        "val_coverage_pct": float(100.0 * ((actual >= lo) & (actual <= hi)).mean()),
        **params,
    }
    return gp, train_bins, valid_bins, summary


def analyze(full, output):
    """Run sensitivity analyses and save talk-ready tables and figures."""
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})

    # 1. Manpower stratification (All vs. Goalie-in-net vs. 5v5)
    strata = {
        "all_situations": lambda d: d,
        "goalie_in_net": lambda d: d[~d["is_goalie_pull"]],
    }
    if "is_5v5" in full.columns:
        strata["five_on_five"] = lambda d: d[d["is_5v5"]]

    stratum_rows = []
    recent_gps = {}
    recent_bins = {}
    valid_all = full[full.era == "validate"]
    for era in ["early", "recent"]:
        era_df = full[full.era == era]
        for sname, fn in strata.items():
            tr = fn(era_df)
            va = fn(valid_all)
            gp, tr_b, va_b, row = evaluate_stratum(tr, va, era, sname)
            stratum_rows.append(row)
            if era == "recent":
                recent_gps[sname] = gp
                recent_bins[sname] = (tr_b, va_b)

    manpower_df = pd.DataFrame(stratum_rows)
    manpower_df.to_csv(output / "manpower_sensitivity.csv", index=False)

    hyper_cols = ["era", "stratum", "constant_norm", "signal_std_xG",
                  "length_scale_s", "noise_level_norm", "residual_std_xG",
                  "log_marginal_likelihood"]
    manpower_df[hyper_cols].to_csv(output / "gp_hyperparameters.csv", index=False)

    # 2. Era-boundary sensitivity across recent-era start years
    train_pool = full[full.role == "train"].copy()
    train_pool["start_year"] = train_pool["season"].map(_season_start_year)
    valid_all_bins = make_bins(valid_all, "validate")
    boundary_rows = []
    for start_year in [2019, 2020, 2021, 2022, 2023]:
        sub = train_pool[train_pool["start_year"] >= start_year]
        tr_bins = make_bins(sub, f"since_{start_year}")
        gp = fit_gp(tr_bins, f"since_{start_year}")
        mean, std = predict_latent(gp, valid_all_bins.midpoint)
        pred_std = predictive_std(gp, std, valid_all_bins.se_sq)
        actual = valid_all_bins.mean_xG.to_numpy()
        lo, hi = mean - 1.96 * pred_std, mean + 1.96 * pred_std
        params = gp_hyperparameters(gp)
        boundary_rows.append({
            "recent_start_year": start_year,
            "recent_label": f"{start_year}-{str(start_year + 1)[-2:]} to 2023-24",
            "n_seasons": int(sub["season"].nunique()),
            "n_shots": int(len(sub)),
            "length_scale_s": params["length_scale_s"],
            "val_rmse": float(np.sqrt(np.mean((actual - mean) ** 2))),
            "val_mae": float(np.abs(actual - mean).mean()),
            "val_coverage_pct": float(100.0 * ((actual >= lo) & (actual <= hi)).mean()),
        })
    boundary_df = pd.DataFrame(boundary_rows)
    boundary_df.to_csv(output / "era_boundary_sensitivity.csv", index=False)

    # 3. Slide 1: Volume vs. Quality decomposition
    pooled_decomp = decompose(train_pool, "2014-15 through 2023-24")
    vol_pct = 100.0 * (pooled_decomp["idx_shots_per_game_per_window_minute"] - 1.0)
    qual_pct = 100.0 * (pooled_decomp["idx_mean_xG"] - 1.0)
    x = np.arange(len(WINDOW_LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - width / 2, vol_pct, width, label="Shot volume (shots / game / min)", color="#4C78A8")
    b2 = ax.bar(x + width / 2, qual_pct, width, label="Shot quality (mean xGoal / shot)", color="#E45756")
    ax.axhline(0, color="black", linewidth=0.8)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.8 if h >= 0 else -1.2
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:+.1f}%",
                ha="center", va=va, fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(WINDOW_LABELS)
    ax.set_ylabel("Change relative to 0–5 min (%)")
    ax.set_title("Trailing by 2 in Period 3: Quality Jumps While Volume Barely Moves")
    ax.set_ylim(min(vol_pct.min(), -5) - 4, qual_pct.max() + 7)
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(figdir / "slide1_volume_vs_quality.png", dpi=300)
    plt.close(fig)

    # 4. Slide 2: Recent-era GP fit + 2024-25 validation
    grid = np.linspace(0, PULL_CUTOFF, 500)
    gp_recent = recent_gps["all_situations"]
    tr_b, va_b = recent_bins["all_situations"]
    mean, std = predict_latent(gp_recent, grid)
    recent_row = manpower_df[(manpower_df.era == "recent") & (manpower_df.stratum == "all_situations")].iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(grid, mean, color="#1f77b4", linewidth=2.4, label="GP latent mean (2022–24)")
    ax.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std, color="#1f77b4", alpha=0.18,
                    label="Pointwise 95% latent interval")
    ax.errorbar(tr_b.midpoint, tr_b.mean_xG, yerr=1.96 * np.sqrt(tr_b.se_sq),
                fmt="o", color="#1f77b4", alpha=0.75, markersize=5, label="Training bins (2022–24)")
    ax.scatter(va_b.midpoint, va_b.mean_xG, color="#d62728", marker="^", s=55, zorder=5,
               label="Held-out 2024–25 bins")
    ax.text(0.03, 0.94,
            f"Length scale: {recent_row.length_scale_s:.0f} s\n"
            f"Held-out RMSE: {recent_row.val_rmse:.5f}\n"
            f"95% PI coverage: {int(round(recent_row.val_coverage_pct * 19 / 100))}/19 bins",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.set(xlim=(0, PULL_CUTOFF), xlabel="Seconds into third period", ylabel="Mean xGoal per shot",
           title="Recent-Era GP Trajectory Generalizes to 2024–25")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(figdir / "slide2_gp_recent_fit.png", dpi=300)
    plt.close(fig)

    # 5. Slide 3: Model comparison on held-out 2024-25 RMSE
    comp_rows = []
    for era in ["early", "recent"]:
        tr_e = make_bins(full[full.era == era], era)
        gp_e = fit_gp(tr_e, era)
        baselines, _ = baseline_predictions(tr_e, valid_all_bins)
        gp_mu, _ = predict_latent(gp_e, valid_all_bins.midpoint)
        actual = valid_all_bins.mean_xG.to_numpy()
        for name, (mu, _) in baselines.items():
            comp_rows.append({"era": era, "model": name, "rmse": float(np.sqrt(np.mean((actual - mu) ** 2)))})
        comp_rows.append({"era": era, "model": "GP (Matérn 5/2)", "rmse": float(np.sqrt(np.mean((actual - gp_mu) ** 2)))})
    comp_df = pd.DataFrame(comp_rows)

    models_order = ["Constant", "Linear", "Exponential", "GP (Matérn 5/2)"]
    y_pos = np.arange(len(models_order))
    h = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    early_vals = [comp_df[(comp_df.era == "early") & (comp_df.model == m)].rmse.iloc[0] for m in models_order]
    recent_vals = [comp_df[(comp_df.era == "recent") & (comp_df.model == m)].rmse.iloc[0] for m in models_order]
    ax.barh(y_pos + h / 2, early_vals, h, label="Trained on Early (2014–22)", color="#9ecae1")
    ax.barh(y_pos - h / 2, recent_vals, h, label="Trained on Recent (2022–24)", color="#1f77b4")
    for idx, (ev, rv) in enumerate(zip(early_vals, recent_vals)):
        ax.text(ev + 0.0002, idx + h / 2, f"{ev:.5f}", va="center", fontsize=8.5)
        ax.text(rv + 0.0002, idx - h / 2, f"{rv:.5f}", va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_order)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out 2024–25 RMSE (lower is better)")
    ax.set_title("Matérn 5/2 GP Outperforms Parametric Baselines Out-of-Sample")
    ax.set_xlim(0, max(early_vals) * 1.2)
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(figdir / "slide3_model_comparison.png", dpi=300)
    plt.close(fig)

    # 6. Slide 4: Manpower stratification + Era-boundary sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    stratum_styles = {
        "all_situations": ("All situations", "#1f77b4", "-"),
        "goalie_in_net": ("Goalie in net (excl. 6v5)", "#ff7f0e", "--"),
        "five_on_five": ("5v5 even strength only", "#2ca02c", "-."),
    }
    for sname, (label, color, ls) in stratum_styles.items():
        if sname in recent_gps:
            m, _ = predict_latent(recent_gps[sname], grid)
            ax1.plot(grid, m, color=color, linestyle=ls, linewidth=2.2, label=label)
    ax1.set(xlim=(0, PULL_CUTOFF), xlabel="Seconds into third period", ylabel="Mean xGoal per shot",
            title="Recent Era: Late Rise Is Driven by Goalie Pulls / Manpower")
    ax1.legend(frameon=False, loc="upper left", fontsize=9)
    _style_axis(ax1)

    ax2.plot(boundary_df.recent_start_year, boundary_df.val_rmse, marker="o", color="#1f77b4", linewidth=2.2)
    chosen = boundary_df[boundary_df.recent_start_year == 2022].iloc[0]
    ax2.scatter([chosen.recent_start_year], [chosen.val_rmse], color="#d62728", s=90, zorder=5,
                label="Chosen split (2022–23)")
    ax2.set_xticks(boundary_df.recent_start_year)
    ax2.set_xticklabels([f"{y}-{str(y + 1)[-2:]}" for y in boundary_df.recent_start_year])
    ax2.set(xlabel="Recent-era start season", ylabel="Held-out 2024–25 RMSE",
            title="Era-Boundary Sensitivity (2024–25 Holdout)")
    ax2.legend(frameon=False, loc="upper right", fontsize=9)
    _style_axis(ax2)

    fig.tight_layout()
    fig.savefig(figdir / "slide4_manpower_and_era.png", dpi=300)
    plt.close(fig)

    print("Saved talk sensitivity tables and 4 slide figures to", output)


def main():
    full, output = run_inputs(__doc__, "talk")
    analyze(full, output)


if __name__ == "__main__":
    main()
