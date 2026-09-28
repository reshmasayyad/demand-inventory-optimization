# Resume and interview notes

A results-supported project description:

> Built a chronological forecast-to-inventory pipeline for 811 products, calibrated residual-based uncertainty, and evaluated five replenishment policies under three supply scenarios; scenario LP allocation reduced weighted simulated cost by 18.1% under tight capacity on a training-selected 30-product cohort.

The 18.1% compares the empirical-buffer LP with empirical-buffer proportional allocation at capacity factor 0.75, under assumed costs. It is not observed company savings. Mention the small aggregate fill-rate tradeoff and that LP does not win every scenario when discussing this result.

## What to understand before an interview

**Why forecast three weeks?** A weekly review with two-week lead time needs a protection-period estimate for lead time plus review interval. Orders placed now arrive two weeks later; the pipeline is included in inventory position.

**Why exclude the supplied normalized columns?** Their MIN/MAX/normalization use the complete time series, including future weeks. The model derives scales from its own training prefix.

**Why use a moving-average baseline?** Additional model complexity is not guaranteed to improve forecasts. The 13-week mean beats boosting on holdout MAE, which is a central result of the study.

**How can a less accurate forecast produce better inventory outcomes?** Absolute-error accuracy and asymmetric inventory losses are different objectives. Forecast bias, stock buffers, initial inventory, capacity and shortage penalties all affect costs. This project reports both dimensions rather than equating them.

**Are the intervals guaranteed?** No. The residual rank follows a conformal-style convention, but serial dependence, overlapping horizons and pooling across products do not establish the exchangeability assumptions. Report empirical class coverage, including undercoverage in B and C.

**What is the LP decision?** How many units of each product to order now, bounded by its maximum request and a common capacity. Scenario shortage/surplus variables make a piecewise expected-cost objective linear. Higher lost-unit penalties represent assumed service priorities.

**Does the LP solve the entire inventory problem optimally?** It solves a continuous myopic surrogate, followed by feasible integer correction. Future inventory evolution and within-lead-time shortages are evaluated by simulation, not optimized globally.

**Why are fill rate and interval coverage different?** Coverage asks whether a three-week quantity is inside a predicted interval or below a target. Fill rate measures the fraction of units served from actual simulated stock. Lead time and prior stock affect fulfillment.

**Is ABC based on real product value?** It uses historical unit volume because prices are absent. A/B/C shortage penalties are scenario assumptions; they are not observed margins.

**How realistic is the demand replay?** Source quantities are a proxy for exogenous demand. Actual availability and hidden lost demand are unknown. The simulator assumes that unfulfilled proxy quantities become known after the week.

**What would improve this?** Longer demand history, multiple selected cohorts, calibration across seasons, availability/censoring information, real costs/capacities, uncertain lead times and multistage optimization.
