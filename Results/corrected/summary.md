# Corrected analysis: 24 September 2026

All 11 configured season files passed input validation. The analysis includes shots taken by the trailing team while down two goals in the third period, from 0:00 through 18:30. Games are identified by season and game ID. The 2025–26 file is outside the study period.

| Sample | Shots | Games | Mean xGoal per shot |
| --- | ---: | ---: | ---: |
| 2014–15 through 2021–22 | 33,252 | 4,690 | 0.06436 |
| 2022–23 through 2023–24 | 9,842 | 1,336 | 0.07236 |
| 2024–25 (held out) | 4,909 | 692 | 0.07220 |

The pooled training sample contains 43,094 shots from 6,026 games. Mean xGoal per shot rises from 0.06174 in the first five minutes to 0.08062 in the final analyzed window (15:00–18:30), a descriptive increase of 30.6%. Shot volume per sampled game per clock-window minute rises by 7.4%. That denominator is not time spent trailing by two.

## Held-out predictions

| Training era | GP RMSE | GP MAE | Bins within nominal 95% prediction interval |
| --- | ---: | ---: | ---: |
| Early | 0.01117 | 0.00917 | 14/19 |
| Recent | 0.00734 | 0.00653 | 17/19 |

The recent-era GP has lower prediction error than the early-era GP and the three recent-era baselines on this held-out season. Coverage is descriptive: 19 correlated bins from one season do not establish calibration. Both GP fits reached the residual white-noise lower bound; sensitivity to that bound remains unchecked.

Shot distance is lower in the recent era in each analysis window. These results do not establish why shot selection changed. The sample includes goalie-absent shots before the cutoff, identified using the trailing team's explicit empty-net flag.

## Outputs

- [Era comparison figure](gp/figures/fig7_gp_comparison.png)
- [Window summaries](decomposition/decomposition_pooled.csv)
- [Training–validation bootstrap differences](decomposition/validation_check.csv)
- [Model comparison](model_comparison/model_comparison.csv)
- [Shot diagnostics](diagnostics/summary_metrics.csv)
- [Input and source fingerprints](run_manifest.json)

Superseded outputs have been removed. All current results are inside this directory.
