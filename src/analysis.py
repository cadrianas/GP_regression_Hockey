"""Shared sample definition and statistics for the comeback analyses."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.paths import DATA_DIR

PULL_CUTOFF = 1110
WINDOW_EDGES = np.array([0, 300, 600, 900, PULL_CUTOFF])
WINDOW_LABELS = ["0–5 min", "5–10 min", "10–15 min", "15–18:30 min"]
WINDOW_MIDPOINTS = (WINDOW_EDGES[:-1] + WINDOW_EDGES[1:]) / 2
WINDOW_MINUTES = np.diff(WINDOW_EDGES) / 60
# MoneyPuck's season field and filename use the season's starting year.
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
    if not np.isfinite(times).all() or (times < edges[0]).any() or (times > edges[-1]).any():
        raise ValueError("Times fall outside the analysis window")
    return np.minimum(np.searchsorted(edges, times, side="right") - 1, len(edges) - 2)


def filter_shots(df, season_year):
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
    if not (df["game_id"] == np.floor(df["game_id"])).all():
        raise ValueError("game_id must be integer-valued")
    home_trails = df["homeTeamGoals"] < df["awayTeamGoals"]
    df["trailingTeam"] = np.where(home_trails, df["homeTeamCode"], df["awayTeamCode"])
    df["is_goalie_pull"] = np.where(home_trails, df["homeEmptyNet"], df["awayEmptyNet"]).astype(bool)
    df["time_in_period"] = df["time"] - 2400
    mask = ((df["period"] == 3)
            & ((df["homeTeamGoals"] - df["awayTeamGoals"]).abs() == 2)
            & (df["teamCode"] == df["trailingTeam"])
            & df["time_in_period"].between(0, PULL_CUTOFF))
    df = df.loc[mask].copy()
    # Keep the raw ID as well as a key that is unique across seasons.
    df["game_key"] = str(season_year) + ":" + df["game_id"].astype("int64").astype(str)
    df["season"] = f"{season_year}-{str(season_year + 1)[-2:]}"
    df["window"] = assign_bins(df["time_in_period"], WINDOW_EDGES)
    return df


def load_dataset(seasons=None):
    """Require the full requested sample; never silently skip a season."""
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


def cluster_bootstrap(df, group_col="window", n_boot=2000, seed=42, groups=None):
    """Shot-weighted means, resampling whole games within each season.

    The same sampled games are used across bins, preserving between-bin
    dependence. A sampled game contributes its original shot sum and count,
    including zero contributions to bins in which it did not shoot.
    """
    if df.empty or n_boot < 2:
        raise ValueError("Bootstrap needs data and at least two replicates")
    groups = np.sort(df[group_col].unique() if groups is None else groups)
    if len(groups) == 0:
        raise ValueError("No populated groups to bootstrap")
    rng = np.random.default_rng(seed)
    sums = np.zeros((n_boot, len(groups)))
    counts = np.zeros_like(sums)
    for _, season in df.groupby("season", sort=True):
        keys = sorted(season["game_key"].unique())
        grouped = season.groupby(["game_key", group_col])["xGoal"].agg(["sum", "count"])
        totals = grouped["sum"].unstack(fill_value=0).reindex(index=keys, columns=groups, fill_value=0).to_numpy()
        sizes = grouped["count"].unstack(fill_value=0).reindex(index=keys, columns=groups, fill_value=0).to_numpy()
        if len(keys) < 2:
            raise ValueError("Bootstrap requires at least two qualifying games per season")
        for start in range(0, n_boot, 100):
            stop = min(start + 100, n_boot)
            draws = rng.integers(len(keys), size=(stop - start, len(keys)))
            sums[start:stop] += totals[draws].sum(axis=1)
            counts[start:stop] += sizes[draws].sum(axis=1)
    samples = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    if np.any(np.isfinite(samples).sum(axis=0) < 0.9 * n_boot):
        raise ValueError("Too few populated bootstrap replicates; use larger bins")
    mean = df.groupby(group_col)["xGoal"].mean().reindex(groups).to_numpy()
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    summary = pd.DataFrame({group_col: groups, "mean_xG": mean, "ci_lo": lo, "ci_hi": hi,
                            "se_sq": np.nanvar(samples, axis=0, ddof=1)})
    return summary.set_index(group_col), samples


def decompose(df, label):
    n_games = df["game_key"].nunique()
    if n_games == 0:
        raise ValueError("Cannot decompose an empty sample")
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
    for col in ["shots_per_game_per_window_minute", "mean_xG", "xG_per_game_per_window_minute"]:
        base = agg.loc[0, col]
        agg[f"idx_{col}"] = agg[col] / base if base > 0 else np.nan
    return agg.reset_index()


def make_bins(df, label, n_boot=2000):
    edges = np.append(np.arange(0, PULL_CUTOFF, 60), PULL_CUTOFF)
    df = df.copy()
    df["bin"] = assign_bins(df["time_in_period"], edges)
    agg = df.groupby("bin")["xGoal"].agg(n="count", mean_xG="mean")
    games = df.groupby("bin")["game_key"].nunique()
    keep = agg.index[(agg["n"] >= 10) & (games >= 2)]
    if len(keep) < 3:
        raise ValueError(f"{label}: need at least three bins with 10 shots and two games")
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
    return np.asarray(variance) / scale**2


def fit_gp(bins_df, label="", n_restarts=10):
    y = bins_df["mean_xG"].to_numpy()
    alpha = normalized_alpha(y, bins_df["se_sq"].to_numpy())
    kernel = (ConstantKernel(1.0, (1e-4, 10.0)) * Matern(200.0, (30.0, 1000.0), nu=2.5)
              + WhiteKernel(1e-4, (1e-6, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                 n_restarts_optimizer=n_restarts, random_state=42)
    return gp.fit(bins_df[["midpoint"]].to_numpy(), y)


def latent_posterior(gp, x):
    """Posterior of the smooth function, excluding WhiteKernel observation noise."""
    x = np.asarray(x).reshape(-1, 1)
    signal = gp.kernel_.k1
    cross = signal(x, gp.X_train_)
    mean = cross @ gp.alpha_ * gp._y_train_std + gp._y_train_mean
    v = solve_triangular(gp.L_, cross.T, lower=True)
    cov = (signal(x) - v.T @ v) * gp._y_train_std**2
    cov = (cov + cov.T) / 2
    return mean, cov


def predict_latent(gp, x):
    mean, cov = latent_posterior(gp, x)
    return mean, np.sqrt(np.maximum(np.diag(cov), 0))


def predictive_std(gp, latent_std, sampling_variance):
    """Uncertainty for a held-out bin mean, including its sampling variance."""
    extra_noise = gp.kernel_.k2.noise_level * gp._y_train_std**2
    return np.sqrt(latent_std**2 + extra_noise + np.asarray(sampling_variance))


def derivative_ratio_samples(gp, times, n_samples=1000, seed=42):
    """Pointwise f'/f intervals from jointly sampled smooth GP trajectories.

    Undefined where fewer than 95% of draws have f > 0.001. These are
    descriptive posterior summaries, not a formal test of exponential shape.
    """
    times = np.asarray(times)
    mean, cov = latent_posterior(gp, times)
    eigval, eigvec = np.linalg.eigh(cov)
    rng = np.random.default_rng(seed)
    draws = mean[:, None] + (eigvec * np.sqrt(np.maximum(eigval, 0))) @ rng.standard_normal((len(times), n_samples))
    derivatives = np.gradient(draws, times, axis=0)
    ratios = np.divide(derivatives, draws, out=np.full_like(draws, np.nan), where=draws > 0.001)
    supported = np.isfinite(ratios).mean(axis=1) >= 0.95
    out = np.full((3, len(times)), np.nan)
    out[:, supported] = np.nanpercentile(ratios[supported], [2.5, 50, 97.5], axis=1)
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
