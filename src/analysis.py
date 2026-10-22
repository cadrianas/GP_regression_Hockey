"""Analyse how shot volume and average shot quality change while a team trails by two.

Select the trailing team's attempts in the third period through 18:30 elapsed.
Shot quality is measured by MoneyPuck's expected-goals value (xGoal).

Summarise the shots in broad time windows, then prepare finer bins for modelling.
Resample whole games within seasons to estimate uncertainty in each bin's mean.
Fit Gaussian Process regression to those means using their estimated variances.

Limitations: the GP fit is not a live game-state model. 
MoneyPuck XG model can change over time.

Aim: separate shot volume and shot quality effects, and describe the shape of the curve.
Could be used as a starting point for late game strategy.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.paths import DATA_DIR

# Using time bins to describe the average shot quality at different points in the third period.
# Elapsed third-period time: 18:30 = 1:30 remaining in a 20-minute period.
# Excluding the final 90 seconds does NOT exclude earlier goalie-pulled shots.
PULL_CUTOFF = 1110
# Broad descriptive windows; the final window is only 3.5 minutes long.
WINDOW_EDGES = np.array([0, 300, 600, 900, PULL_CUTOFF])
WINDOW_LABELS = ["0-5 min", "5-10 min", "10-15 min", "15-18:30 min"]
WINDOW_MIDPOINTS = (WINDOW_EDGES[:-1] + WINDOW_EDGES[1:]) / 2
WINDOW_MINUTES = np.diff(WINDOW_EDGES) / 60
# MoneyPuck's season field and filename use the season's starting year.
# The recent-versus-earlier split is exploratory, not an estimated change point.
# The 2024-start file is later-season evaluation, never passed to GP fitting.
# Eras season 2014-2015 to 2021-2022 `early`, 2022-2023, 2023-2024 `recent`, 2024-2025 `validate`.
    
SEASONS = {
    DATA_DIR / f"shots_{year}.csv": {
        "year": year,
        "label": f"{year}-{str(year + 1)[-2:]}",
        "era": "early" if year <= 2021 else "recent" if year <= 2023 else "validate",
        "role": "train" if year <= 2023 else "validate",
    }
    for year in range(2014, 2025)
}
EARLY_SEASONS = [m["label"] for m in SEASONS.values() if m["era"] == "early"]
RECENT_SEASONS = [m["label"] for m in SEASONS.values() if m["era"] == "recent"]
REQUIRED_COLUMNS = {
    "season", "game_id", "period", "time", "homeTeamGoals", "awayTeamGoals",
    "homeTeamCode", "awayTeamCode", "teamCode", "xGoal", "homeEmptyNet", "awayEmptyNet",
}


def assign_bins(times, edges):
    """Left-closed bins, including the analysis cutoff in the last bin."""
    times = np.asarray(times, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError("Bin edges must be finite and strictly increasing")
    if not np.isfinite(times).all() or (times < edges[0]).any() or (times > edges[-1]).any():
        raise ValueError("Times fall outside the analysis window")
    return np.minimum(np.searchsorted(edges, times, side="right") - 1, len(edges) - 2)

#Function that keeps the trailing team's shots in the third period through 18:30 elapsed, 
# and adds analysis identifiers.
def filter_shots(df, season_year):
    """Return one season's eligible shot rows with analysis identifiers.

    Keep attempts by the team down exactly two, in period three, through 18:30
    elapsed inclusive. Eligibility is evaluated at each attempt, so games can
    enter or leave the sample as the score changes. All manpower situations
    remain eligible; goalie absence is recorded rather than filtered out.
    """
    # Fail on malformed inputs rather than silently changing the study sample.
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required shot columns: {', '.join(sorted(missing))}")
    df = df.copy()
    numeric = ["season", "game_id", "period", "time", "homeTeamGoals", "awayTeamGoals",
               "xGoal", "homeEmptyNet", "awayEmptyNet"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values in {col}")
    if set(df["season"].unique()) != {season_year}:
        raise ValueError(f"Expected season {season_year}, found {sorted(df['season'].unique())}")
    if not df["xGoal"].between(0, 1).all():
        raise ValueError("xGoal must be between zero and one")
    for col in ["homeEmptyNet", "awayEmptyNet"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain 0/1 flags")
    for col in ["game_id", "period", "homeTeamGoals", "awayTeamGoals"]:
        if not (df[col] == np.floor(df[col])).all():
            raise ValueError(f"{col} must be integer-valued")
        if (df[col] < (1 if col in ["game_id", "period"] else 0)).any():
            raise ValueError(f"Invalid negative or zero value in {col}")
    for col in ["homeTeamCode", "awayTeamCode", "teamCode"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing team identity in {col}")
    if df.homeTeamCode.eq(df.awayTeamCode).any():
        raise ValueError("Home and away teams must differ")
    if not (df.teamCode.eq(df.homeTeamCode) | df.teamCode.eq(df.awayTeamCode)).all():
        raise ValueError("Shooting team must match the home or away team")
    # MoneyPuck shotID can repeat between games, so use the compound key.
    if "shotID" in df and df.duplicated(["game_id", "shotID"]).any():
        raise ValueError("Duplicate (game_id, shotID) in a season")
    
    # Identify who is trailing using the score BEFORE the shot.
    # A goal scored while down two still qualifies, even though it reduces the deficit.
    home_trails = df["homeTeamGoals"] < df["awayTeamGoals"]
    df["trailingTeam"] = np.where(home_trails, df["homeTeamCode"], df["awayTeamCode"])
    # Record whether the trailing team's goalie is absent.
    # Keep this flag for diagnostics; it does not exclude shots from the sample.
    df["is_goalie_pull"] = np.where(home_trails, df["homeEmptyNet"], df["awayEmptyNet"]).astype(bool)
    # Convert elapsed game time to elapsed third-period time.
    # Subtract the first two periods: 40 minutes = 2400 seconds.
    df["time_in_period"] = df["time"] - 2400
    # Focus on down two: a situation where a comeback still felt possible.
    # Keep only shots BY the trailing team, not shots by the team ahead.
    # All four conditions must hold. Period 3 excludes overtime.
    # The time range includes both 0:00 and 18:30 elapsed (1:30 remaining).
    mask = ((df["period"] == 3)
            & ((df["homeTeamGoals"] - df["awayTeamGoals"]).abs() == 2)
            & (df["teamCode"] == df["trailingTeam"])
            & df["time_in_period"].between(0, PULL_CUTOFF))
    df = df.loc[mask].copy()
    # Combine season and game ID so games are distinct across seasons.
    # The bootstrap uses this key to redraw whole games.
    df["game_key"] = str(season_year) + ":" + df["game_id"].astype("int64").astype(str)
    df["season"] = f"{season_year}-{str(season_year + 1)[-2:]}"
    df["window"] = assign_bins(df["time_in_period"], WINDOW_EDGES)
    return df


def load_dataset(seasons=None):
    """Require the full requested sample; never silently skip a season.
    Eras season 2014-2015 to 2021-2022 `early`, 2022-2023, 2023-2024 `recent`, 2024-2025 `validate`. """
    seasons = SEASONS if seasons is None else seasons
    missing = [str(p) for p in seasons if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing required season files:\n  " + "\n  ".join(missing)
                                + "\nDownload the configured seasons from https://moneypuck.com/data.htm")
    frames = []
    for path, meta in seasons.items():
        df = filter_shots(pd.read_csv(path), meta["year"])
        if df.empty:
            raise ValueError(f"No qualifying shots in {path}")
        df["role"], df["era"] = meta["role"], meta["era"]
        frames.append(df)
    if not frames:
        raise ValueError("No seasons configured")
    return pd.concat(frames, ignore_index=True)

# Calculate mean xG by time bin and estimate its uncertainty by resampling
# whole games within each season. Return means (total xG / total shots), intervals, and variances of means.
def cluster_bootstrap(df, group_col="window", n_boot=2000, seed=42, groups=None):
    """Shot-weighted means, resampling whole games within each season.

    Shots from one game share circumstances, resample games to preserve that clustering. 
    The target is the mean over shots, not an equal-weight mean of games. 
    
    Returns a per-bin summary (original mean, percentile interval, variance of
    resampled means) and joint draws of shape (n_boot, number_of_bins). The
    target is the mean over selected shots, not an equal-weight mean of games.
    Only games represented in the input sample are resampled.
    """
    if df.empty or n_boot < 2:
        raise ValueError("Bootstrap needs data and at least two replicates")
    groups = np.sort(df[group_col].unique() if groups is None else groups)
    if len(groups) == 0:
        raise ValueError("No populated groups to bootstrap")
    rng = np.random.default_rng(seed)
    # One row per repetition and one column per bin; accumulate across seasons.
    sums = np.zeros((n_boot, len(groups)))
    counts = np.zeros_like(sums)
    # Stratify by season: each repetition draws its observed number of games.
    for _, season in df.groupby("season", sort=True):
        keys = sorted(season["game_key"].unique())
        # Sufficient statistics avoid copying every shot for every repetition.
        # Both matrices have shape (games_in_season, number_of_bins).
        # A game with no attempts in a bin contributes zero xG and zero count.
        # Grouped array has xG totals and shot counts
        grouped = season.groupby(["game_key", group_col])["xGoal"].agg(["sum", "count"])
        totals = grouped["sum"].unstack(fill_value=0).reindex(index=keys, columns=groups, fill_value=0).to_numpy()
        sizes = grouped["count"].unstack(fill_value=0).reindex(index=keys, columns=groups, fill_value=0).to_numpy()
        if len(keys) < 2:
            raise ValueError("Bootstrap requires at least two qualifying games per season")
        # Batching limits temporary memory; it does not change the estimator.
        for start in range(0, n_boot, 100):
            stop = min(start + 100, n_boot)
            # Each row selects games with replacement. Repeated IDs contribute
            # repeatedly. Reuse these indices for sums, counts, AND every bin.
            draws = rng.integers(len(keys), size=(stop - start, len(keys)))
            sums[start:stop] += totals[draws].sum(axis=1)
            counts[start:stop] += sizes[draws].sum(axis=1)
    # Each redraw: total xG / total shots gives the shot-weighted bin mean.
    # Leave NaN when a redraw has no shots in that bin.
    samples = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    # Require a defined mean in at least 90% of redraws for every bin.
    if np.any(np.isfinite(samples).sum(axis=0) < 0.9 * n_boot):
        raise ValueError("Too few populated bootstrap replicates; use larger bins")
    # Report the mean from the original shots, not the average of bootstrap means.
    mean = df.groupby(group_col)["xGoal"].mean().reindex(groups).to_numpy()
    # Middle 95% of resampled means gives the bootstrap interval.
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    # se_sq measures uncertainty in the estimated mean, not variation among shots.
    summary = pd.DataFrame({group_col: groups, "mean_xG": mean, "ci_lo": lo, "ci_hi": hi,
                            "se_sq": np.nanvar(samples, axis=0, ddof=1)})
    return summary.set_index(group_col), samples


def decompose(df, label):
    """Summarise volume, total xG, and mean quality in broad clock windows.

    The game denominator includes represented games with no shots in a given
    window, but not games with no eligible shots anywhere in this input sample.
    Counts are normalised by clock-window duration, not score-state exposure.
    """
    n_games = df["game_key"].nunique()
    if n_games == 0:
        raise ValueError("Cannot decompose an empty sample")
    # Use the same shot rows so total_xG = shots * mean_xG in nonempty bins.
    agg = df.groupby("window")["xGoal"].agg(
        shots="count", total_xG="sum", mean_xG="mean", median_xG="median", std_xG="std"
    ).reindex(range(4))
    agg[["shots", "total_xG"]] = agg[["shots", "total_xG"]].fillna(0)
    agg["shots_per_game"] = agg["shots"] / n_games
    agg["xG_per_game"] = agg["total_xG"] / n_games
    agg["window_minutes"] = WINDOW_MINUTES
    # Clock-window normalization, NOT a rate per minute spent trailing by two.
    agg["shots_per_game_per_window_minute"] = agg["shots_per_game"] / WINDOW_MINUTES
    agg["xG_per_game_per_window_minute"] = agg["xG_per_game"] / WINDOW_MINUTES
    agg["n_games"], agg["season"] = n_games, label
    agg["window_label"], agg["midpoint_s"] = WINDOW_LABELS, WINDOW_MIDPOINTS
    # Dimensionless indices compare relative changes from the first window.
    for col in ["shots_per_game_per_window_minute", "mean_xG", "xG_per_game_per_window_minute"]:
        base = agg.loc[0, col]
        agg[f"idx_{col}"] = agg[col] / base if base > 0 else np.nan
    return agg.reset_index()

# Need finner bins to fit a smooth curve
def make_bins(df, label, n_boot=2000):
    """Build the GP input table: bin mean, sampling variance, and time midpoint.

    Use minute-wide bins plus a final 30-second bin through 18:30. A bin mean
    is represented at its midpoint, an approximation to within-bin variation.
    Each bin has at least 10 shots and at least two games; otherwise it is dropped.
    We have 19 observations for the GP fit.
    """
    edges = np.append(np.arange(0, PULL_CUTOFF, 60), PULL_CUTOFF)
    df = df.copy()
    df["bin"] = assign_bins(df["time_in_period"], edges)
    agg = df.groupby("bin")["xGoal"].agg(n="count", mean_xG="mean")
    games = df.groupby("bin")["game_key"].nunique()
    # Require both shot and game support; shot count alone can hide clustering.
    keep = agg.index[(agg["n"] >= 10) & (games >= 2)]
    if len(keep) < 3:
        raise ValueError(f"{label}: need at least three bins with 10 shots and two games")
    # Reuse the whole-game bootstrap at this finer temporal resolution.
    # The GP receives marginal bin variances, not full cross-bin covariance.
    summary, _ = cluster_bootstrap(df, "bin", n_boot=n_boot, groups=keep)
    agg = agg.loc[keep].copy()
    agg["se_sq"] = summary.loc[keep, "se_sq"]
    agg["midpoint"] = (edges[keep] + edges[keep + 1]) / 2
    agg["label"] = label
    return agg.reset_index()


def normalized_alpha(y, variance):
    """scikit-learn normalizes y but does not rescale alpha for the caller."""
    scale = np.std(y)
    if scale < 10 * np.finfo(float).eps:
        scale = 1.0
    # Variances have squared units; do not divide by shot count again here.
    return np.asarray(variance) / scale**2

# GP regression models time as a smooth latent function plus independent residual noise. 
# The GP is fit to the observed means, not individual shot outcomes. 
def fit_gp(bins_df, label="", n_restarts=10, noise_bounds=(1e-6, 1.0)):
    """Fit a smooth mean-xG trajectory using each bin's estimated precision.
    Target is the mean over shots in a bin, not individual shot outcomes. The
    GP is a smooth latent function plus independent residual noise. 

    Input: the table returned by make_bins. Output: a fitted scikit-learn GP.
    Hyperparameters are optimised by marginal likelihood.
    Subsequent function intervals condition on these fitted values and alpha.
    n_restarts controls how many random initialisations are tried for the hyperparameter search.
    """
    # Observed averages
    # Each row is an estimated bin mean, not an individual shot outcome.
    # Noisier means need not pull the fitted curve as closely toward themselves.
    # Sampling uncertainty of the observed means, in the normalised y units.
    y = bins_df["mean_xG"].to_numpy()
    
    # Estimated uncertainity of the observed means, in the normalised y units.
    alpha = normalized_alpha(y, bins_df["se_sq"].to_numpy())

    #Kernerl: Smooth signal + additional independent residual variance. WhiteKernel complements 
    # the supplied bootstrap variance.
    # The Matern length scale controls how quickly the curve can vary in time.
    # Smoothing is intentional: the fit need not pass through every bin mean.
    kernel = (ConstantKernel(1.0, (1e-4, 10.0)) * Matern(200.0, (30.0, 1000.0), nu=2.5)
              + WhiteKernel(np.clip(1e-4, *noise_bounds), noise_bounds))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                 n_restarts_optimizer=n_restarts, random_state=42)
    # Time is the sole predictor. Neither validation responses nor individual
    # shot outcomes enter this fit; there is no imposed increasing trend.
    return gp.fit(bins_df[["midpoint"]].to_numpy(), y)


def latent_posterior(gp, x):
    """Posterior of the smooth function, excluding WhiteKernel observation noise."""
    x = np.asarray(x).reshape(-1, 1)
    # This helper assumes the fitted structure (smooth signal + WhiteKernel).
    # Keep training noise in L_, but omit new-observation noise at query times.
    signal = gp.kernel_.k1
    cross = signal(x, gp.X_train_)
    mean = cross @ gp.alpha_ * gp._y_train_std + gp._y_train_mean
    # Use the fitted Cholesky factor rather than explicitly invert covariance.
    v = solve_triangular(gp.L_, cross.T, lower=True)
    cov = (signal(x) - v.T @ v) * gp._y_train_std**2
    cov = (cov + cov.T) / 2
    return mean, cov


def predict_latent(gp, x):
    """Return smooth-function means and pointwise standard deviations."""
    mean, cov = latent_posterior(gp, x)
    return mean, np.sqrt(np.maximum(np.diag(cov), 0))


def predictive_std(gp, latent_std, sampling_variance):
    """Uncertainty for a held-out bin mean, including its sampling variance."""
    # Convert residual variance back to original xG units before adding it.
    # Three terms: latent variance + residual variance + held-out mean variance.
    extra_noise = gp.kernel_.k2.noise_level * gp._y_train_std**2
    return np.sqrt(latent_std**2 + extra_noise + np.asarray(sampling_variance))


def derivative_ratio_samples(gp, times, n_samples=1000, seed=42):
    """Describe the curve's rate of change relative to its level.

    For a positive exponential curve, f'/f would be constant.
    These pointwise summaries are exploratory, not a formal test.
    """
    times = np.asarray(times)

    # Get the smooth curve's mean and joint uncertainty across times.
    mean, cov = latent_posterior(gp, times)

    # Decompose the covariance to generate curves with correlated time points.
    eigval, eigvec = np.linalg.eigh(cov)
    rng = np.random.default_rng(seed)

    # Each column is one sampled curve, conditional on the fitted GP settings.
    # Clip negative eigenvalues to zero before taking square roots.
    draws = mean[:, None] + (
        eigvec * np.sqrt(np.maximum(eigval, 0))
    ) @ rng.standard_normal((len(times), n_samples))

    # Estimate each sampled curve's slope along the time axis.
    derivatives = np.gradient(draws, times, axis=0)

    # Divide slope by curve level; avoid unstable division near zero.
    ratios = np.divide(
        derivatives, draws,
        out=np.full_like(draws, np.nan),
        where=draws > 0.001,
    )

    # Report a time point only if at least 95% of draws give valid ratios.
    supported = np.isfinite(ratios).mean(axis=1) >= 0.95

    # Return lower limit, median, and upper limit at each supported time.
    # Unsupported times remain NaN; these are pointwise intervals.
    out = np.full((3, len(times)), np.nan)
    out[:, supported] = np.nanpercentile(
        ratios[supported], [2.5, 50, 97.5], axis=1
    )
    return out


def run_inputs(description, subdirectory):
    """Common CLI; check all inputs before creating or replacing any outputs."""
    import argparse
    from src.paths import RESULTS_DIR

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / subdirectory)
    args = parser.parse_args()
    seasons = {args.data_dir / p.name: meta for p, meta in SEASONS.items()}
    try:
        full = load_dataset(seasons)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return full, args.output_dir
