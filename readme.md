# hockeybayes: From Pressure Dynamics to Strategic Adaptation


## The Research Journey

This project began with a simple question and revealed something far larger. Here's what happened.

### Phase 1: The Within-Period Question

**Original hypothesis:** When an NHL team trails by two goals in the third period, does **pressure and time scarcity** cause them to take higher-quality shots?

Using 47,046 shots from 2014–2024 in comeback situations (down 2 goals, 3rd period), we binned shots into four five-minute windows and found:

| Window | Mean xGoal | % Change |
|--------|-----------|----------|
| 0–5 min (early) | 0.0640 | — |
| 5–10 min | 0.0661 | +3.1% |
| 10–15 min | 0.0653 | +2.0% |
| **15–20 min (late)** | **0.0940** | **+46.9%** |

**Result:** Highly significant (t = 21.06, p < 1e-97). Shot quality spiked in the final five minutes.

**Interpretation:** Within a single game situation, teams do improve shot quality under desperation. But this told only half the story.

### Phase 2: Pooling Assumption Breaks

When we tested whether this pattern was **stable across seasons**, something unexpected happened:

- We ran **Kruskal-Wallis tests** separately for each time window
- **Result:** Massive heterogeneity. Early seasons (2014–18) looked different from recent seasons (2022–24)
- **Insight:** We couldn't just pool all years—the **underlying dynamics had shifted**

Shot quality in comeback situations wasn't just driven by *time pressure within a game*. It was driven by something **structural that changed across the decade**.

### Phase 3: The Era Story Emerges

We split the data into two eras:

- **Early Era (2014–21):** 33,252 shots across 5,862 games
- **Recent Era (2022–24):** 9,842 shots across 1,705 games

And decomposed expected goals per game:

$$\text{xG}_\text{per game} = \text{Shot Volume} \times \text{Shot Quality}$$

**Finding:**

| Metric | Early (2014–21) | Recent (2022–24) | Change |
|--------|-----------------|------------------|--------|
| Shots per game | 1.97 | 1.98 | **+0.5%** (no change) |
| Mean xGoal per shot | 0.0582 | 0.0654 | **+12.4%** |
| **Total xG per game** | **0.1146** | **0.1295** | **+13.0%** |

**The mechanism:** Shot volume stayed flat. All improvement came from **higher-quality shot selection**—teams deliberately moved shots toward higher-danger locations (slot vs. perimeter).

**Mechanism details:**
- Danger-zone concentration (slots < 20 ft): **27% → 31%** (+5 pp)
- Slapshot frequency: **12.2% → 6.8%** (−5.4 pp)
- Net-front tips and snaps: Increased substantially

This is not random variation. This is **strategic optimization** in response to analytics maturity.

---

## The Narrative: Analytics Adoption Changed Behavior

The NHL's analytics timeline:

- **2010–2015:** Pioneering teams (Toronto, Winnipeg) develop xGoal models
- **2015–2020:** Gradual league-wide adoption; xGoal becomes standard metric
- **2020–2022:** COVID acceleration; analytics legitimacy solidifies
- **2022–2024:** Mature infrastructure achieved → shift from *measurement* to *optimization*

**We observe a structural break around 2022–23.** This is when teams stopped just measuring shot quality and started *optimizing strategy around it*.

The evidence:
- Trailing teams explicitly reposition toward higher-danger zones
- This happens consistently across all five-minute windows (not desperation-driven)
- The pattern persists in validation (2024–25 held-out season)
- Effect size is substantial: +12.4% quality without volume increases

**Conclusion:** Quantitative insights into win probability modified real in-game behavior. Teams learned to generate better shots under time pressure—and they learned to do it systematically, across the full period, not just in final moments.

---

## What This Package Contains

This repository provides the **statistical tools and reproducible analysis** behind the paper:

### Analysis Pipeline

1. **Data loading and filtering** — Load MoneyPuck shot-level CSVs, filter to comeback situations
2. **Temporal binning** — Aggregate shots into 5-minute windows for within-period analysis
3. **Decomposition analysis** — Separate volume and quality drivers using indexed metrics
4. **Heterogeneity testing** — Kruskal-Wallis tests to validate pooling assumptions
5. **Era-level comparison** — Stratify by season period (early vs. recent) and diagnose mechanisms
6. **Bayesian modeling** — Gaussian Process regression to smooth trajectories and quantify uncertainty
7. **Publication figures** — Generate press-ready plots with credible bands and annotations

### Key Output Artifacts

- **Decomposition table** — Volume, quality, and total xG per game by era
- **Heterogeneity test results** — P-values and test statistics for pooling assumption
- **Gaussian Process posterior** — Mean trajectory + 95% credible intervals
- **Diagnostic plots** — Within-period dynamics, era comparison, shot-type composition
- **Validation results** — Out-of-sample predictions vs. 2024–25 held-out season

---

## Methodological Notes

### Why Decomposition First?

Before fitting complex models, we decompose to understand **what changed**: volume or quality? This reveals the mechanism and guides subsequent analysis.

### Why Heterogeneity Testing?

Pooling years together assumes they're equivalent. Kruskal-Wallis tests validate that assumption. If violated (as we found), it signals structural breaks that warrant era-level analysis.

### Why Gaussian Process?

The GP makes no parametric assumption about trajectory shape. The posterior provides calibrated credible intervals that preserve uncertainty across the entire temporal domain. This is critical for a paper: reviewers want to know *not just* the point estimate, but the confidence around it.

### Why Matérn 5/2?

- Assumes the target function is twice-differentiable
- More appropriate than RBF for a domain subject to discrete events (line changes, penalties, strategic shifts)
- Computationally efficient
- Theoretically justified for smooth athletic dynamics

### Why Game-Level Bootstrap?

Shots within a game are correlated (same goalie, ice conditions, opponent). Shot-level bootstrap ignores this and underestimates uncertainty. Game-level bootstrap preserves correlation structure and produces honest confidence intervals.

---

## Limitations

Understanding what this analysis **does not** do is as important as understanding what it does.

### Descriptive, Not Causal

The decomposition and heterogeneity tests show **associations** between era and shot quality. They do not establish causation. The observed improvement is consistent with analytics adoption, but other confounds could exist (roster changes, rule changes, defensive evolution). The paper argues that the *timing* and *mechanism* (location-based, not volume-based) make analytics adoption the most plausible explanation, but causality requires caution.

### League-Wide Aggregation Masks Team Heterogeneity

All analyses pool across 32 NHL teams. Individual teams may exhibit substantially different adaptation patterns depending on coaching philosophy, analytics investment, and roster stability. The league-wide pattern is a macro finding; team analytics staffs should treat it as a prior, not a prescription.

### Era Boundary Is a Design Choice

We split at 2022–23, motivated by institutional analytics timelines. The boundary is somewhat arbitrary. Sensitivity analysis (results with alternative boundaries: 2021, 2023) should be performed.

### Goalie-Pull Regime Excluded

Shots after 1110 seconds (~18:30 into the period) are excluded. This avoids contaminating the urgency signal with the structural regime change of goalie removal, but it means the analysis says nothing about the 6-on-5 period—arguably the highest-stakes window of a comeback.

### Validation on Partial Season

The 2024–25 validation set has ~5k shots (partial season). Conclusions should be interpreted with the smaller sample in mind. Full-season validation is deferred until the season completes.

### xGoal Uncertainty Not Propagated

xGoal values are treated as fixed, even though they are themselves model predictions from MoneyPuck's expected goals model. Systematic bias in that model (e.g., underestimating danger-zone xGoal) would propagate into our estimates. This package takes xGoal as given and does not propagate model uncertainty.

---

## Citation

If you use this analysis in research, please cite the companion paper:

```bibtex
@article{ciupeanu2025adaptation,
  title   = {Strategic Adaptation in High-Leverage Situations: 
             Shot Selection Evolution in {NHL} Comeback Attempts},
  author  = {Ciupeanu, Adriana-Stefania},
  journal = {Journal of Quantitative Analysis in Sports},
  year    = {2025},
  note    = {Preprint: arXiv:XXXX.XXXXX}
}
```

---

## Key Takeaways

| Question | Finding | Evidence |
|----------|---------|----------|
| **Do teams take better shots under pressure?** | Yes, within a single game situation, shot quality increases 46.9% from early to late period | t-test: p < 1e-97, n=22.5k shots |
| **Is this pattern stable over time?** | No. Early seasons (2014–21) behave differently from recent seasons (2022–24) | Kruskal-Wallis: p < 0.0001 for most windows |
| **What drove the era-level improvement?** | Shot quality increased 12.4% between eras, but volume stayed flat | Decomposition: quality 0.0582 → 0.0654 |
| **How did quality improve without more shots?** | Teams moved shots toward higher-danger zones (slots vs. perimeter, tips vs. slaps) | Location analysis: danger zone 27% → 31% |
| **Is this deliberate or random variation?** | Deliberate and systematic—effect is uniform across all 5-min windows and persists in validation | Heterogeneity test + out-of-sample validation |

---
  
**Last Updated:** May 2026
**author** Adriana-Stefania Ciupeanu
**licence** Code licence under GNU General Public License v3.0 (GPLv3)