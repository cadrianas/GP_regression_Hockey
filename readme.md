# hockeybayes

**Nonparametric Bayesian tools for NHL shot quality analysis.**

`hockeybayes` provides a clean pipeline for studying within-period shot quality
dynamics in comeback situations using Gaussian Process regression. It is the
companion package to:

> Ciupeanu, A.-S. (2025). *Does Pressure Change How Teams Play? Nonparametric
> Estimation of Within-Period Shot Quality Dynamics in NHL Comeback Situations.*


---

## What it does

When an NHL team trails by two goals in the third period, does their shot
quality improve as time runs out — and if so, does it rise gradually or spike
at the end? This package provides the statistical tools to answer that question
rigorously.

The core finding from the companion paper: **shot quality, not shot volume,
drives late-period expected goal growth in comeback situations**. Mean xGoal per
shot increases 47.8% between the opening and closing five-minute windows, while
shot volume *declines* by 17.1%. The temporal structure of this increase is not
exponential — it is a flat plateau for the first 15 minutes followed by a sharp
threshold effect in the final three minutes of regulation.

---

## Installation

```bash
pip install hockeybayes
```

**Requirements:** Python ≥ 3.9, numpy, pandas, scikit-learn, scipy, matplotlib.

---

## Quick start

```python
import hockeybayes as hb

# Load and filter MoneyPuck shot data
# Download from: https://moneypuck.com/data.htm
SEASON_MAP = {
    "Data/shots_2023.csv": {"label": "2022-23", "role": "train"},
    "Data/shots_2024.csv": {"label": "2023-24", "role": "train"},
    "Data/shots_2025.csv": {"label": "2024-25", "role": "validate"},
}

train, valid = hb.load_multiple_seasons(SEASON_MAP)

# Decompose into volume and quality components
table = hb.decompose(train)
print(table[["window_label", "shots_per_game", "mean_xG", "xG_per_game"]])

# Fit a Gaussian Process to the within-period quality trajectory
model = hb.fit_period_gp(train)

# Test whether an exponential model is consistent with the GP posterior
result = hb.exponential_consistency_test(model)
print(f"Exponential inside GP 95% CI: {result.pct_inside:.1f}%")

# Identify the inflection point and peak rate of change
deriv = hb.derivative_analysis(model)
print(f"Inflection at t={deriv.t_inflection:.0f}s ({deriv.t_inflection/60:.1f} min)")
print(f"Peak rate: {deriv.peak_rate_per_min:+.5f} xGoal/shot/min")

# Plot
fig, ax = hb.plot_posterior(model)
fig, ax = hb.plot_exponential_overlay(model, result)
fig, axes = hb.plot_derivative(model, deriv)
```

---

## Data

This package uses shot-level CSV data from
[MoneyPuck](https://moneypuck.com/data.htm). Download
`shots_{YEAR}.csv` files (where `YEAR` is the season-ending year) and place
them in a `Data/` directory. The package handles filtering automatically.

**MoneyPuck naming convention:**  
`shots_2024.csv` = 2023–24 season (season value `2024` in the data)

**Columns required:**  
`period`, `homeTeamGoals`, `awayTeamGoals`, `homeTeamCode`, `awayTeamCode`,
`teamCode`, `time`, `xGoal`, `game_id`

---

## API reference

### Loading data

```python
# Load a single season
shots = hb.load_season("Data/shots_2024.csv", season_label="2023-24")

# Load multiple seasons and split into train / validate
train, valid = hb.load_multiple_seasons(SEASON_MAP)

# Apply the filter to an already-loaded DataFrame
filtered = hb.comeback_filter(df, season_label="2023-24", role="train")
```

**Comeback filter logic:**
- Third period only (`period == 3`)
- Exactly two-goal deficit (`|homeGoals − awayGoals| == 2`)
- Trailing team's shots only
- Excludes `time_in_period > 1110s` (goalie-pull regime)

### Decomposition

```python
# Four-window decomposition: shots × mean xGoal = total xGoal
table = hb.decompose(train)

# Game-level bootstrap confidence intervals (respects within-game correlation)
ci_table = hb.bootstrap_ci(train, n_boot=2000, ci=0.95)

# Plain-English summary of dominant driver
summary = hb.volume_quality_summary(train)
# summary["dominant_driver"]      → "QUALITY" or "VOLUME"
# summary["quality_change_pct"]   → float
# summary["coaching_implication"] → str
```

### Gaussian Process regression

```python
# Fit GP with Matérn 5/2 kernel and heteroscedastic noise
model = hb.fit_period_gp(train, bin_size=60, pull_cutoff=1110)

# Predict posterior at arbitrary time points
mu, ci_lo, ci_hi = hb.predict_trajectory(model, t=np.linspace(0, 1110, 500))

# Test exponential consistency
result = hb.exponential_consistency_test(model)
# result.pct_inside        → float: % of grid inside 95% CI
# result.early_failure_pct → float: early-region failure %
# result.late_failure_pct  → float: late-region failure %
# result.detail            → pd.DataFrame: per-point breakdown

# Numerical derivative analysis
deriv = hb.derivative_analysis(model)
# deriv.t_inflection        → float: inflection point in seconds
# deriv.t_peak_rate         → float: time of maximum rate of change
# deriv.peak_rate_per_min   → float: xGoal/shot/minute at peak
```

### Plotting

All plot functions return `(fig, ax)` for full customisation.

```python
hb.plot_posterior(model)                        # GP mean + CI + training bins
hb.plot_exponential_overlay(model, result)      # Exponential consistency test
hb.plot_derivative(model, deriv)                # Rate of change (two panels)
hb.plot_validation(model, valid_bins)           # Validation vs training CI
hb.plot_ci_width(model, deriv)                  # Uncertainty over time
```

---

## Methodological notes

**Why Gaussian Process regression?**  
GP regression makes no parametric assumption about the shape of the
within-period quality trajectory. The posterior distribution over functions
provides calibrated credible intervals that can be used to formally test whether
a specific parametric model (e.g., exponential growth) is consistent with the
data — a test that has no natural analogue under rolling-average smoothers.

**Why Matérn 5/2?**  
The Matérn 5/2 kernel assumes the target function is twice-differentiable.
This is more appropriate than the infinitely smooth RBF kernel for a domain
subject to discrete events (penalties, line changes, goalie pulls) that
create genuine, if smooth, structural changes in shot dynamics.

**Why game-level bootstrap?**  
Shots within the same game are correlated — same goalie, same ice conditions,
same score state. Shot-level bootstrap underestimates uncertainty by treating
50 shots from one game as 50 independent observations. Game-level bootstrap
preserves this correlation structure and produces honest confidence intervals.

**Goalie-pull exclusion:**  
Shots with `time_in_period > 1110s` are excluded to avoid contaminating the
urgency signal with the structural regime change that follows goalie removal.
The goalie-pull epoch can be analysed separately by adjusting `pull_cutoff`.

---

## Extending the package

The comeback filter accepts arbitrary `goal_diff` values, so the same pipeline
applies to one-goal deficits, three-goal deficits, or other game states:

```python
# One-goal deficit situations
shots_1g = hb.load_season("Data/shots_2024.csv", season_label="2023-24",
                           goal_diff=1)

# Different period
# (requires modifying time_in_period offset — period 2 starts at t=1200s)
```

---

---

## Limitations

Understanding what this package does *not* do is as important as understanding
what it does. The following limitations apply to both the methodology and the
scope of the companion paper.

**This is a descriptive, not causal, analysis.**  
The GP estimates the functional form of the association between time into the
third period and mean xGoal per shot. It does not establish that the passage
of time *causes* shot quality to rise. The observed pattern is consistent with
several causal mechanisms — tactical shifts by trailing teams, fatigue-induced
defensive breakdowns, line-matching decisions by the leading team — but the
data cannot distinguish between them. Any causal interpretation requires
additional evidence beyond what this package provides.

**League-wide aggregation obscures team-level heterogeneity.**  
All analyses pool shots across all 32 NHL teams. Individual teams may exhibit
substantially different within-period shot quality trajectories depending on
coaching philosophy, roster composition, and opponent strength. A team analytics
staff applying these findings should treat the league-wide pattern as a prior,
not a prescription, and estimate team-specific effects separately. The package
currently has no facility for team-level GP fitting; this is a planned
extension.

**The goalie-pull epoch is excluded, not modelled.**  
Shots after `time_in_period > 1110s` (approximately 18:30 of the third) are
excluded to avoid contaminating the urgency signal with the structural regime
change that follows goalie removal. This exclusion means the package says
nothing about the 6-on-5 period — arguably the highest-stakes window of any
comeback situation. The goalie-pull epoch warrants its own analysis with an
explicit changepoint model; the current GP is not designed for it.

**The two-goal deficit filter is a design choice, not a universal finding.**  
Results are conditioned on trailing teams facing exactly a two-goal deficit.
The threshold effect may be attenuated for one-goal deficits (where
trailing teams face less urgency early) or absent for three-goal deficits
(where comebacks are rare enough that tactical behaviour may differ
qualitatively). The `goal_diff` parameter makes it straightforward to
re-run the analysis for other deficit sizes, but the findings should not
be generalised beyond the conditions studied without separate validation.

**xGoal models carry their own uncertainty.**  
The xGoal values used as the outcome variable are themselves model predictions
from MoneyPuck's expected goals model — not ground truth shot danger. Any
systematic bias in that model (e.g., underestimating shot quality from certain
locations or shot types) propagates into the GP estimates. This package takes
xGoal as given and does not propagate xGoal model uncertainty into the
posterior. Users applying results to a specific team's shot data should be
aware that the findings are only as reliable as the underlying xGoal model.

**Binning discards shot-level structure.**  
The GP is fitted to 60-second bin means rather than individual shots. This
approach is computationally efficient and statistically transparent, but it
aggregates away within-bin variation in shot timing and quality. A shot-level
hierarchical GP — modelling individual xGoal values as draws from a latent
function with game-level and team-level random effects — would be more
principled but substantially more complex to implement and interpret. The
binned approach is appropriate for the research question addressed here but
should not be assumed to capture all relevant structure in the data.

**Validation is based on a partial season.**  
The 2024–25 validation season contains 3,649 shots across 527 games, compared
to approximately 5,000 shots per full training season. The smaller sample
increases bin-level variance and reduces the power of the posterior predictive
check. The 78.9% GP coverage reported in the companion paper should be
interpreted with this caveat in mind; full-season validation is planned once
the 2024–25 data is complete.

## Citation

If you use this package in research, please cite the companion paper:

```bibtex
@article{ciupeanu2025pressure,
  title   = {Does Pressure Change How Teams Play? Nonparametric Estimation
             of Within-Period Shot Quality Dynamics in {NHL} Comeback Situations},
  author  = {Ciupeanu, Adriana-Stefania},
  journal = {Journal of Quantitative Analysis in Sports},
  year    = {2025},
  note    = {Preprint: arXiv:XXXX.XXXXX}
}
```

---

