"""Regression tests for sample definition, weighting, and GP uncertainty."""
import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from src.analysis import (SEASONS, WINDOW_EDGES, assign_bins, cluster_bootstrap,
                          decompose, derivative_ratio_samples, filter_shots, fit_gp,
                          latent_posterior, load_dataset, make_bins, normalized_alpha,
                          predictive_std)
from src.decomposition import compare_samples
from src.diagnostic_shot_location import goalie_summary
from src.results import baseline_predictions


def raw_shot(**overrides):
    row = dict(season=2023, game_id=20001, period=3, time=2500,
               homeTeamGoals=0, awayTeamGoals=2, homeTeamCode="A", awayTeamCode="B",
               teamCode="A", xGoal=.1, homeEmptyNet=0, awayEmptyNet=0,
               homeSkatersOnIce=5, awaySkatersOnIce=4)
    row.update(overrides)
    return row


def clustered_sample():
    # Game IDs deliberately collide across two seasons, and shot counts differ.
    rows = []
    for year in [2022, 2023]:
        for game, n, quality in [(1, 1, .1), (2, 9, .9)]:
            for w in range(4):
                for _ in range(n):
                    rows.append(dict(season=str(year), game_key=f"{year}:{game}", game_id=game,
                                     window=w, xGoal=quality, time_in_period=[100, 400, 700, 1100][w]))
    return pd.DataFrame(rows)


class SampleTests(unittest.TestCase):
    def test_repeated_game_ids_remain_separate(self):
        a = filter_shots(pd.DataFrame([raw_shot()]), 2023)
        b = filter_shots(pd.DataFrame([raw_shot(season=2024)]), 2024)
        self.assertEqual(pd.concat([a, b]).game_key.nunique(), 2)
        result = decompose(clustered_sample(), "test")
        self.assertTrue((result.n_games == 4).all())
        self.assertAlmostEqual(result.shots_per_game.iloc[0], 5)

    def test_goalie_flags_not_score_or_skater_advantage(self):
        shots = pd.DataFrame([raw_shot(), raw_shot(homeEmptyNet=1),
                              raw_shot(homeTeamGoals=2, awayTeamGoals=0, teamCode="B", awayEmptyNet=1)])
        result = filter_shots(shots, 2023)
        self.assertEqual(result.is_goalie_pull.tolist(), [False, True, True])
        self.assertAlmostEqual(goalie_summary(result).goalie_pull_shot_pct.iloc[0], 200 / 3)

    def test_boundaries_and_only_trailing_shots(self):
        rows = [raw_shot(time=t) for t in [2399, 2400, 2700, 3000, 3300, 3510, 3511]]
        rows += [raw_shot(teamCode="B"), raw_shot(period=2), raw_shot(homeTeamGoals=1)]
        result = filter_shots(pd.DataFrame(rows), 2023)
        self.assertEqual(result.window.tolist(), [0, 1, 2, 3, 3])
        self.assertEqual(assign_bins([0, 300, 1110], WINDOW_EDGES).tolist(), [0, 1, 3])

    def test_bad_source_data_is_rejected(self):
        for overrides in [dict(season=2022), dict(xGoal=np.nan), dict(xGoal=1.2), dict(homeEmptyNet=2)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                filter_shots(pd.DataFrame([raw_shot(**overrides)]), 2023)
        with self.assertRaisesRegex(ValueError, "Missing required"):
            filter_shots(pd.DataFrame([raw_shot()]).drop(columns="awayEmptyNet"), 2023)

    def test_missing_files_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "shots_2014.csv"):
                load_dataset({Path(temp) / "shots_2014.csv": next(iter(SEASONS.values()))})

    def test_short_window_normalization(self):
        result = decompose(clustered_sample(), "test")
        self.assertEqual(result.window_minutes.tolist(), [5, 5, 5, 3.5])
        self.assertEqual(result.midpoint_s.iloc[-1], 1005)
        self.assertAlmostEqual(result.shots_per_game_per_window_minute.iloc[-1], 5 / 3.5)

    def test_bootstrap_preserves_shot_weight_and_bin_dependence(self):
        summary, draws = cluster_bootstrap(clustered_sample(), n_boot=500)
        self.assertTrue(np.allclose(summary.mean_xG, .82))  # Equal-game mean would be .5.
        np.testing.assert_allclose(draws[:, 0], draws[:, 3])
        # Outcomes retain each of the four game clusters, rather than two merged IDs.
        self.assertGreater(len(np.unique(draws[:, 0].round(8))), 3)
        _, repeat = cluster_bootstrap(clustered_sample(), n_boot=500)
        np.testing.assert_allclose(draws, repeat)

    def test_comparison_uses_same_estimand_and_both_samples(self):
        sample = clustered_sample()
        result = compare_samples(sample, sample, n_boot=500)
        np.testing.assert_allclose(result.train_mean_xG, result.valid_mean_xG)
        self.assertTrue((result.difference_ci_lo < 0).all())
        self.assertTrue((result.difference_ci_hi > 0).all())

    def test_last_gp_bin_midpoint_respects_cutoff(self):
        bins = make_bins(clustered_sample(), "test", n_boot=100)
        self.assertEqual(bins.midpoint.iloc[-1], 1095)

    def test_sparse_gp_bins_are_dropped_before_bootstrapping(self):
        sample = clustered_sample()
        rare = sample.iloc[[0]].copy()
        rare["time_in_period"] = 200
        bins = make_bins(pd.concat([sample, rare]), "test", n_boot=100)
        self.assertNotIn(3, bins.bin.tolist())
        self.assertEqual(len(bins), 4)

    def test_sparse_bootstrap_reports_problem(self):
        with self.assertRaisesRegex(ValueError, "two qualifying games"):
            cluster_bootstrap(clustered_sample().query("game_id == 1"), n_boot=100)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.bins = pd.DataFrame({"midpoint": np.linspace(30, 1095, 8),
                                  "mean_xG": [.06, .065, .063, .065, .07, .073, .085, .10],
                                  "se_sq": np.full(8, 1e-5)})

    def test_alpha_normalization_matches_manually_scaled_fit(self):
        gp = fit_gp(self.bins, n_restarts=0)
        y = self.bins.mean_xG.to_numpy()
        scaled = (y - y.mean()) / y.std()
        manual = GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None,
                                           alpha=self.bins.se_sq.to_numpy() / y.var(), normalize_y=False)
        manual.fit(self.bins[["midpoint"]].to_numpy(), scaled)
        a, sa = gp.predict([[200], [600]], return_std=True)
        b, sb = manual.predict([[200], [600]], return_std=True)
        np.testing.assert_allclose(a, b * y.std() + y.mean())
        np.testing.assert_allclose(sa, sb * y.std())
        np.testing.assert_allclose(normalized_alpha(np.ones(4), np.ones(4)), np.ones(4))

    def test_latent_covariance_excludes_white_noise(self):
        gp = fit_gp(self.bins, n_restarts=0)
        query = np.array([100, 500, 900])
        mean, cov = latent_posterior(gp, query)
        observed, noisy_cov = gp.predict(query[:, None], return_cov=True)
        np.testing.assert_allclose(mean, observed)
        noise = gp.kernel_.k2.noise_level * gp._y_train_std**2
        np.testing.assert_allclose(noisy_cov - cov, np.eye(3) * noise, atol=1e-12)
        expected = np.diag(cov) + noise + .001
        np.testing.assert_allclose(predictive_std(gp, np.sqrt(np.diag(cov)), .001)**2, expected)

    def test_derivative_uncertainty_with_constant_marginal_variance(self):
        t = np.linspace(0, 100, 30)
        cov = .0001 * np.exp(-((t[:, None] - t[None, :]) / 30)**2)
        with patch("src.analysis.latent_posterior", return_value=(np.full(30, .1), cov)):
            low, median, high = derivative_ratio_samples(None, t, n_samples=1000)
        self.assertTrue(np.isfinite(median).all())
        self.assertTrue(((high - low) > 0).all())
        self.assertTrue(((low < 0) & (high > 0)).all())

    def test_baseline_linear_fit_and_covariance(self):
        data = self.bins.copy()
        data["mean_xG"] = .05 + .00003 * data.midpoint
        predictions, _ = baseline_predictions(data, data)
        mean, variance = predictions["Linear"]
        np.testing.assert_allclose(mean, data.mean_xG, atol=1e-12)
        self.assertTrue((variance > 0).all())


class EntryPointTests(unittest.TestCase):
    def test_imports_do_not_run_analyses(self):
        with patch("pandas.read_csv", side_effect=AssertionError("Import read data")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                for name in ["GP_regression", "decomposition", "diagnostic_shot_location",
                             "empty_net_frequency", "results", "Diagnostic"]:
                    importlib.reload(importlib.import_module("src." + name))
            self.assertEqual(output.getvalue(), "")

    def test_cli_missing_inputs_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            process = subprocess.run([sys.executable, "-m", "src.GP_regression", "--data-dir", temp,
                                      "--output-dir", str(output)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn("Missing required season files", process.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
