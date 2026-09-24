# hockeybayes

Gaussian process regression for NHL shot quality when a team is down two goals in the third period.

Does a team generate better chances as time runs out? This project uses MoneyPuck shot data to examine how expected goals per shot change through the period, and whether that pattern holds across seasons. The question is about the chances teams create while trailing, rather than whether they eventually complete a comeback.

The analysis starts with shot counts and average shot quality, then fits Gaussian processes to the time trajectories. Separating those quantities matters: a team can generate more expected goals by shooting more often, taking better shots, or both.

## Approach

The main scripts keep shots taken by the trailing team when the score difference is exactly two goals in the third period. They use MoneyPuck's `xGoal` value as the measure of shot quality and exclude shots after 18:30 of the period.

There are three parts to the analysis:

1. **Decomposition.** Compare shot volume, mean xGoal, and total xG in five-minute windows. Exploratory ANOVA and Kruskal–Wallis tests compare game-mean distributions across seasons, with a correction for inspecting four windows. These tests use a different estimand from the shot-weighted summaries.
2. **Gaussian process regression.** Fit separate trajectories for the earlier and more recent seasons using one-minute averages. The model uses a Matérn 5/2 kernel with a constant scale and white noise, plus sampling variances estimated by resampling whole games within each season. The variances are rescaled to match the normalized target.
3. **Diagnostics and validation.** Examine shot locations and shot types, and compare the fitted trajectories with a held-out season.

The current GP configuration assigns 2014–15 through 2021–22 to the early group, 2022–23 through 2023–24 to the recent group, and 2024–25 to validation. Filenames use the starting year: `shots_2024.csv` contains 2024–25. The loader checks the source `season` field against this mapping.

## Reading the results

The saved analyses show higher average xGoal late in the period and differences across seasons. Those patterns motivate fitting separate era trajectories instead of treating the whole decade as one sample.

They do not establish that time pressure causes better shot selection, or that analytics adoption explains the changes between seasons. Shot quality also depends on personnel, defensive play, game state, and the mix of shots that enter the sample. Deliberate tactical adaptation is one possible explanation, but this analysis cannot distinguish it from those alternatives.

The analysis was rerun on September 2026 using all 11 configured season files (first analysis only used 3 seasons, then it was updated to include from the 2014-2015 season up to the 2025-2026 season). [Current results](Results/corrected/summary.md) include 43,094 training shots from 6,026 games and 4,909 held-out shots from 692 games. Mean xGoal rises from 0.06174 in the first window to 0.08062 in the last analyzed window, a descriptive increase of 30.6%. The recent-era GP has lower held-out RMSE than the early-era fit (0.00734 versus 0.01117); its nominal 95% prediction intervals cover 17 of 19 validation bins.

Current tables, figures, and a run manifest are in `Results/corrected/`. Superseded results, notebooks, and model artifacts have been removed.

## Running the analysis

From the repository root, create a Python environment and install the libraries used by the main scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the shot-level CSVs from [MoneyPuck](https://moneypuck.com/data.htm). The main scripts look for `shots_2014.csv` through `shots_2024.csv` in the directory set by `DATA_DIR` in `src/paths.py`. The path uses an existing `Data/` directory, or falls back to `data/`. Each analysis accepts `--data-dir` and `--output-dir` to override the defaults.

Season and era definitions live in `src/analysis.py`. The analysis commands require all configured files and stop with a list of missing inputs before writing outputs. `shots_2025.csv` is outside the configured study period.

```bash
python src/decomposition.py --output-dir Results/corrected/decomposition
python src/GP_regression.py --output-dir Results/corrected/gp
python src/diagnostic_shot_location.py --output-dir Results/corrected/diagnostics
python src/results.py --output-dir Results/corrected/model_comparison
```

These scripts read the shot data independently and write tables and figures under `Results/`. Raw season files are not included in Git. Dependencies are pinned to the versions used for verification with Python 3.9.

## Repository guide

| Path | Contents |
| --- | --- |
| `src/decomposition.py` | Window summaries, season comparisons, and bootstrap validation |
| `src/GP_regression.py` | Separate GP fits by era and held-out comparisons |
| `src/diagnostic_shot_location.py` | Shot-location and shot-type diagnostics |
| `src/analysis.py` | Shared filters, season definitions, bootstrap, and GP functions |
| `src/paths.py` | Shared data and output paths |
| `Results/corrected/` | Current tables, figures, results summary, and run manifest |
| `tests/` | Regression tests for filtering, weighting, and model uncertainty |
| `requirements.txt` | Pinned analysis dependencies |
| `gaussian_process_lecture_notes.md` | Background notes on Gaussian processes |

`src/results.py` now reads the configured shot files directly and compares each era's GP with constant, linear, and exponential baselines. The commands above write its outputs to `Results/corrected/model_comparison/`. Derivative-ratio intervals come from jointly sampled latent GP trajectories; they are descriptive, not a formal test rejecting exponential behavior.

Run the regression tests with:

```bash
python -m unittest discover -s tests -v
```

To inspect the available local season files without requiring the complete study sample:

```bash
python src/Diagnostic.py
```

## Limits and unfinished work

- The 18:30 cutoff does not exclude every pulled-goalie situation. The diagnostics use the trailing team's explicit empty-net flag and report a percentage of sampled shots, not games or playing time. The final window is labeled 15–18:30; volume comparisons are normalized by clock-window duration. They are not shooting rates per minute spent trailing by two, which would require game-state exposure data.
- Whole games are resampled within seasons, preserving shot weights and joint bin movements. The GP uses the resulting marginal bin variances but does not model their full covariance. Its latent intervals are pointwise and conditional on fitted hyperparameters. Held-out prediction intervals additionally include validation sampling variance and fitted residual noise.
- Game keys combine season and game ID. Per-game summaries describe games with at least one qualifying shot; games with no qualifying shots are not represented in that denominator. Training–validation difference intervals include bootstrap uncertainty from both samples.
- The era boundary is a modeling choice. Differences between seasons do not, by themselves, identify a structural break in 2022–23.
- xGoal is itself a model estimate. This analysis treats it as fixed and does not propagate uncertainty from MoneyPuck's model.
- League-wide averages can hide differences between teams. Team-level comparisons and sensitivity to alternative era boundaries remain useful next steps.

## Author and acknowledgements

Adriana-Stefania Ciupeanu

Code license: GNU General Public License v3.0 (GPLv3), as stated for this project. A standalone license file still needs to be added.

Grammarly and Google Jules were used during the project.
