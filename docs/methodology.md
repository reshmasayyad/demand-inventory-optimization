# Methodology and mathematical assumptions

## Chronological fit and forecasting

The prediction target at origin t is the raw quantity summed over weeks t, t+1 and t+2. Initial models observe W0–W23. Their training examples use origins 13–21, so every target is available by W23. Selection origins 24–27 have targets ending by W29. The chosen boosting complexity is refitted using W0–W29, with training origins 13–27. Calibration origins 30–36 require labels through W38. Holdout origins 39–49 have fully observed targets through W51.

All 811 products enter the forecasting comparison, including inactive/intermittent products. A pooled model uses scale-normalized lag 1/2/4 quantities, rolling means over 4/8/13 weeks, rolling standard deviation and zero frequency, recent trend, log training scale and training-only ABC class. Numeric scale is max(training mean, 1). Scales are fitted on each model's training prefix and are frozen for its forecasts; they are not calculated from the complete 52-week series.

Two boosting configurations, 15 and 31 leaves, use squared-error loss, 120 iterations, learning rate 0.06, L2 regularization 3 and seed 42. Random internal early stopping is disabled. The 15-leaf model has lower selection-period MAE within this model family. Baselines are reported alongside it; the selected model is not claimed to beat all baselines. In the holdout, the 13-week mean has MAE 4.302, compared with 6.097 for boosting.

Provided MIN, MAX and Normalized fields are excluded because their computation uses the full history. Product IDs identify rows but are not numeric model features. ABC classes use W0–W23 volume shares, approximately 80%/15%/5%, and remain fixed after refitting.

## Residual calibration

Signed normalized residuals equal (actual protection-period quantity − prediction) / training scale. Residuals are pooled by frozen ABC class across seven calibration origins. Absolute residuals provide symmetric interval radii; signed residuals provide upper buffers, floored at zero. Quantiles use rank ceil((n+1) × 0.90). If the rank exceeds the number of calibration scores, the quantile is infinite rather than claiming information that is unavailable.

This convention resembles split conformal prediction, but no standard distribution-free guarantee is asserted. The source is serially dependent, horizons overlap, products share week effects, and class pooling can obscure conditional heterogeneity. Empirical symmetric holdout coverage is 91.25% for A, 87.84% for B, and 85.31% for C. Upper-target coverage is 87.10%, 83.46% and 82.25%, respectively. These are distinct from weekly inventory fill rates.

## Operational cohort and costs

Inventory simulation uses ten initially active products from each ABC class, sampled with seed 42 before the test period is seen. The 30-product cohort is a class-balanced operational sample, not a volume-representative estimate for the complete dataset.

Review period R=1 week; replenishment lead L=2 weeks; protection horizon L+R=3. Capacity per review is ceil(factor × sum of cohort means estimated from W0–W29), with factors 0.75, 1.00 and 1.25. Inventory costs are hypothetical: end-week holding 1 per unit; lost-unit penalties A=20, B=10, C=5. They reflect assumed service priorities derived from volume classes, not observed prices or margins. Procurement and setup costs are excluded.

All policies start at W39 with ceil(2 × refit mean) units on hand per product and no pipeline. Each week: receive orders due now; compute inventory position; place capacity-limited orders; expose realized source quantities; fulfill available units; discard unfulfilled units as lost sales; charge holding and shortage costs. Lost demand does not become a backlog. Orders placed at t arrive before demand at t+2.

All policies observe the same source demand proxy after the week, including units lost in the hypothetical system. Actual sales censoring and missing demand are not inferred. The comparison is a replay with uncensored proxy feedback, not a causal simulation of how customer demand responds to stockouts.

Orders that would arrive after W51 are suppressed for every policy. This common finite-horizon rule ensures terminal pipeline is zero, while terminal on-hand stock remains reported. Initial stock and the short terminal horizon may influence policy rankings.

## Policies and allocation

Mean-only targets three times the trailing 13-week mean. The normal-buffer benchmark adds z_0.90 × sqrt(3) × trailing standard deviation; its independence/normality assumptions are not verified. Boost-only uses the point protection-period forecast. Empirical-buffer uses the calibrated signed-residual upper target.

Requests are max(ceil(target) − inventory position, 0). For proportional allocation, distribute capacity in proportion to requests and use largest fractional remainders to produce feasible integer units. The LP policy uses the same maximum requests but may choose smaller orders.

For every LP review, generate 20 nonnegative protection-period demand scenarios using evenly spaced quantiles of class-pooled signed residuals, multiplied by each product's training scale and added to its forecast. With inventory position IP, order q, scenario demand D, shortage u and surplus v:

```text
minimize mean_scenario(sum_product(shortage_penalty * u + holding_penalty * v))
subject to IP + q + u - v = D for every product/scenario
           sum(q) <= shared_capacity
           0 <= q <= requested_units
           u, v >= 0
```

This is a sparse continuous LP with a separable expected-cost objective and one shared unit-capacity constraint. No claim about a realistic joint demand distribution is needed for this marginal expectation surrogate, although a joint chance-constrained model would require such information.

SciPy/HiGHS solves the relaxation. Floor its order vector and greedily add feasible units while they reduce the marginal scenario objective. This is a feasible integer correction; no universal integer-optimality claim is made. Tests compare a small problem against exhaustive integer enumeration. The integer scenario objective is audited against the relaxation lower bound; the largest observed rounding gap is approximately 2.0944 surrogate cost units.

The LP is myopic. It treats inventory position as available over the protection horizon and does not model all within-lead-time shortage timing, multiweek state transitions or ordering fixed costs. Its expected objective is not the same as the simulator's cumulative realized cost. A lower LP objective does not guarantee the best realized dynamic inventory policy.

## Reported outcomes

Forecast MAE/RMSE concern three-week totals. WAPE sums absolute forecast errors and divides by summed observed quantities; no percentage errors are calculated separately for zero-demand items. Overlapping targets create dependence, so no independent-observation significance claim is made.

Unit fill rate = fulfilled units / proxy demand units. Item-week no-stockout rate = fraction of product-weeks with zero lost units. Relevant realized cost = sum(end-week holding units + class-weighted lost units). Terminal stock and total orders are also reported. Costs are abstract units and are not claimed as retailer profit or actual savings.

At capacity factor 0.75, empirical-buffer proportional allocation costs 9,661 with 87.87% fill; LP costs 7,913 with 87.66% fill. The 18.1% weighted-cost reduction is specific to that cohort, horizon, capacity and penalty scenario. At factor 1.00, boost-only has slightly lower relevant cost than LP; the empirical buffer has the highest aggregate fill rate. Policy rankings are not universal.

Longer datasets, multiple training-only cohorts, a simulation warm-up, demand censoring treatment, asymmetric economic forecast objectives, rolling calibration and a dynamic optimization model would strengthen the study.
