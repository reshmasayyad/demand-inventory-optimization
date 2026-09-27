from pathlib import Path
import nbformat as n
r=Path(__file__).resolve().parents[1]
c=[]
def md(s):c.append(n.v4.new_markdown_cell(s.strip()))
def code(s):c.append(n.v4.new_code_cell(s.strip()))
md(r'''# Demand Forecasting and Inventory Optimization

**Question:** How do forecast uncertainty and shared supply capacity change replenishment decisions, service and cost?

We forecast three-week purchase quantities, compare statistical and ML point forecasts, calibrate residual-based intervals, and simulate five inventory policies. A scenario linear program allocates a limited common supply budget across a training-selected 30-product cohort.

Data: [Sales Transactions Weekly, UCI](https://doi.org/10.24432/C5XS4Q), CC BY 4.0. This is an offline operational case study with assumed costs and lead times, not an evaluation of a retailer's actual inventory system.''')
code('''import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
ROOT=Path.cwd()
if not (ROOT/'src').exists(): ROOT=ROOT.parent
assert (ROOT/'src/demand_inventory').exists(), 'Open this notebook within the extracted repository.'
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Image
from demand_inventory.data import load_sales, product_metadata, select_inventory_cohort
from demand_inventory.pipeline import run
from demand_inventory.reporting import LABELS, COLORS, pareto, forecast_comparison, coverage_plot, cost_panels, service_cost_panels
plt.rcParams.update({'axes.spines.top':False, 'axes.spines.right':False, 'font.size':10})
FIGURES=ROOT/'reports/figures'; FIGURES.mkdir(parents=True, exist_ok=True)
def show(fig, name):
    path=FIGURES/f'{name}.png'; fig.savefig(path, dpi=150, bbox_inches='tight')
    display(Image(filename=str(path))); plt.close(fig)''')
md(r'''## 1. Data audit and availability

The source has 811 products with raw quantities W0–W51. It supplies no calendar dates, stock availability, prices, lead times or capacities. We retain week indices and treat recorded purchases as an exogenous demand proxy. Zero records remain zero quantities.

**Leakage trap:** the supplied MIN, MAX and Normalized columns summarize all 52 weeks. We exclude them and fit scales using the observed training prefix only. Lag and rolling features exclude the forecast-origin week.

The simulator reveals each week's demand proxy after ordering. All policies receive the same proxy history, including units the simulated system fails to fulfill. This assumes uncensored demand feedback; it does not infer hidden demand from the source retailer's censored sales.''')
code('''codes, values=load_sales(ROOT/'data/raw/Sales_Transactions_Dataset_Weekly.csv')
initial=product_metadata(codes, values, cutoff=24)
cohort=select_inventory_cohort(initial, each_class=10, seed=42)
print('Products:',len(codes),'| Weeks:',values.shape[1],'| Product-week records:',values.size)
print('Zero recorded product-weeks:',int((values==0).sum()))
print('Products inactive in W0–W23:',int((initial.training_total==0).sum()))
display(initial.groupby('abc').agg(products=('product','size'), initial_units=('training_total','sum'), mean_zero_fraction=('zero_fraction','mean')).round(3))
assert values.shape==(811,52)
assert len(cohort)==30
assert initial.iloc[cohort].abc.value_counts().to_dict()=={'A':10,'B':10,'C':10}''')
code('''show(pareto(initial),'01_training_abc_pareto')''')
md(r'''## 2. Forecast the protection period

Review every week, with an assumed two-week replenishment lead time. The order-up-to policy considers the **lead time plus review period**, so the forecast target at origin $t$ is

$$D^{(3)}_{i,t}=d_{i,t}+d_{i,t+1}+d_{i,t+2}.$$

Chronology is explicit:

| Stage | Observations or origins | Last target available |
|---|---|---|
| Initial model fit | Observe W0–W23 | W23 |
| Select boosting complexity | Origins 24–27 | W29 |
| Refit chosen complexity | Observe W0–W29 | W29 |
| Calibrate uncertainty | Origins 30–36 | W38 |
| Evaluate three-week forecasts | Origins 39–49 | W51 |
| Simulate inventory decisions | W39–W51 | Each week, after ordering |

Baselines repeat the last observed week or use the trailing 4-/13-week mean. A pooled histogram gradient boosting model predicts normalized three-week totals using lags, rolling means/variation, zero frequency, a recent-trend feature, training scale and ABC class. We choose between 15 and 31 leaves using selection-period MAE **within the boosting family**; baselines are reported alongside it. The exercise does not assume boosting is the best overall forecaster.

The frozen refit model and calibration remain unchanged throughout the holdout, while lag features update with newly observed weeks.''')
code('''result=run(ROOT, save=True)
print('Selected boosting leaves:',result['configuration']['selected_max_leaf_nodes'])
display(result['selection_scores'].round(3))
display(result['test_forecast_scores'].round(3))
print('Test origins × products per model:',11*811)
assert result['calibration'].last_label_week==38
assert result['model'].fit_cutoff==30
show(forecast_comparison(result['test_forecast_scores']),'02_forecast_accuracy')''')
md(r'''### Forecast interpretation

The 13-week mean has lower holdout MAE than boosting. Preserve that finding: additional model complexity is not automatically useful. MAE and WAPE evaluate three-week totals over all 811 products; they are not annual revenue or one-week forecasting metrics. Overlapping targets and common week effects mean these errors are dependent, so no independent-observation significance test is asserted.''')
md(r'''## 3. Calibrate and check uncertainty

Normalize signed residuals by each product's training scale $s_i=\max(\bar d_i,1)$ and pool them within frozen ABC classes. For a symmetric interval, take the finite-sample rank quantile of absolute calibration residuals. For an upper order target, take the corresponding quantile of signed residuals, truncated below at zero:

$$[\max(0,\hat D-r_gs_i),\ \hat D+r_gs_i],\qquad S_i=\max(0,\hat D+b_gs_i).$$

The rank convention is $\lceil(n_g+1)0.90\rceil$. This resembles split conformal calibration, but exchangeability is not established for serial, overlapping and cross-product observations. **We report empirical coverage; we do not promise 90% deployment coverage.** Thousands of product-origin residuals come from only seven distinct calibration origins.

The two-sided interval and one-sided upper target are different objects. Their demand-total coverage is also different from weekly unit fill rate in the inventory simulation.''')
code('''coverage=result['interval_coverage']
display(coverage.round(3))
intervals=result['test_intervals']
print('Overall symmetric interval coverage:',round(float(intervals.covered.mean()),3))
print('Overall upper-target coverage:',round(float(intervals.upper_covered.mean()),3))
print('Normalized absolute radii:',result['calibration'].radius)
print('Normalized one-sided buffers:',result['calibration'].upper_buffer)
show(coverage_plot(coverage),'03_interval_coverage')''')
code('''meta=result['inventory_cohort']
example=meta.loc[meta.abc=='A'].sort_values('mean').iloc[5]['product']
group=intervals.loc[intervals['product']==example].sort_values('origin')
fig,ax=plt.subplots(figsize=(9,4))
ax.fill_between(group.origin.to_numpy(),group.lower.to_numpy(),group.upper.to_numpy(),color='#bfdbfe',alpha=.8,label='Empirical 90% symmetric interval')
ax.plot(group.origin,group.actual,'o-',color='#0f172a',label='Observed three-week quantity')
ax.plot(group.origin,group.prediction,'o-',color='#2563eb',label='Boosting forecast')
ax.set(title=f'Product {example}: rolling three-week totals',xlabel='Forecast origin (week index)',ylabel='Units')
ax.legend(fontsize=8);fig.tight_layout();show(fig,'04_example_forecast_interval')''')
md(r'''## 4. Inventory simulator

At the start of week $t$: receive due orders; compute inventory position (on hand plus pipeline); choose an order; then observe demand, fulfill available units, record lost sales, and charge end-week holding cost. An order placed at $t$ arrives before demand at $t+2$.

Five policies:

| Policy | Order-up-to target | Shared-capacity allocation |
|---|---|---|
| Mean only | Three times the trailing 13-week mean | Proportional |
| Normal buffer | Mean plus $z_{0.90}\sqrt3$ times trailing standard deviation | Proportional |
| Boost only | Frozen boosting point forecast | Proportional |
| Empirical buffer | Boosting forecast plus calibrated upper buffer | Proportional |
| Buffer + LP | Same maximum target | Scenario LP |

The normal buffer assumes an approximate independent normal weekly-demand model; it is a benchmark rather than a verified probabilistic description.

All policies begin with the same on-hand state: two weeks of refit-period mean demand, rounded upward, with no pipeline. End-of-horizon orders that would arrive after W51 are suppressed. Terminal stock is reported. This short finite-horizon initialization can affect results.

Assumed costs: holding = 1 per unit/week; lost-unit penalties A=20, B=10, C=5. ABC reflects recorded volume, not monetary value; penalties are hypothetical service priorities. Cost units are abstract. Acquisition costs, ordering setup costs and actual retailer profit are not modeled.

Weekly capacity is 0.75, 1.00 or 1.25 times the selected cohort's refit-period mean demand, rounded to units. This gives **15 policy/capacity cases**, each evaluated on 30 products × 13 weeks = 390 item-weeks.''')
code('''summary=result['inventory_summary']
display(summary[['capacity_factor','policy','total_cost','fill_rate','item_week_no_stockout_rate','lost_units','terminal_on_hand']].round(3))
assert len(summary)==15
assert (summary.evaluated_item_weeks==390).all()
assert (summary.terminal_pipeline==0).all()
show(cost_panels(summary),'05_inventory_cost_components')''')
md(r'''## 5. Allocate limited supply with a linear program

For each review, represent protection-period demand using 20 class-residual quantile scenarios. Let $IP_i$ be inventory position, $q_i$ the order, and $u_{ik},v_{ik}$ scenario shortage and surplus. Solve

$$\min\frac1K\sum_{i,k}(p_i u_{ik}+h_i v_{ik})$$

subject to

$$IP_i+q_i+u_{ik}-v_{ik}=D_{ik},\quad\sum_i q_i\le C,\quad 0\le q_i\le r_i,\quad u_{ik},v_{ik}\ge0,$$

where $r_i$ is the requested replenishment up to the empirical upper target. Sparse matrices and SciPy's HiGHS solver compute the continuous relaxation. Floor the solution and add feasible units while they reduce the scenario objective. The integer allocation's objective is audited against the relaxed lower bound.

This optimizes a **myopic ending-stock/shortage surrogate**, not a multistage globally optimal lost-sales policy. Inventory position includes future receipts, so the LP cannot fully represent within-lead-time timing of shortages. It can leave capacity unused if another unit increases the surrogate cost. Joint-scenario dependence is not validated; the separable expected-cost objective uses marginal residual scenarios.''')
code('''audit=result['allocation_audit']
print('Successful LP decisions:',len(audit))
print('Largest integer correction gap:',round(float(audit.rounding_gap.max()),4),'surrogate cost units')
assert (audit.integer_objective>=audit.lp_objective-1e-6).all()
records=result['inventory_records']
weekly=records.groupby(['capacity_factor','policy','week']).agg(ordered=('ordered','sum'),capacity=('capacity','first'),requested=('requested','sum')).reset_index()
assert (weekly.ordered<=weekly.capacity).all()
assert (records.ordered<=records.requested).all()
assert (records.demand==records.fulfilled+records.lost).all()
assert (records.ending_on_hand>=0).all()
print('Capacity, requested-order bounds, demand accounting and LP lower bounds verified.')
show(service_cost_panels(summary),'06_service_cost_tradeoff')''')
md(r'''## 6. Examine class priorities and inventory flows

An aggregate fill rate counts every unit equally. Weighted shortage costs emphasize class A. An allocation can reduce weighted cost while lowering total units fulfilled; compare both measures instead of treating one as a substitute for the other.''')
code('''class_rows=[]
for (factor,policy,abc),g in records.groupby(['capacity_factor','policy','abc']):
    class_rows.append({'capacity_factor':factor,'policy':policy,'abc':abc,'demand':int(g.demand.sum()),
                       'fill_rate':float(g.fulfilled.sum()/g.demand.sum()) if g.demand.sum() else 1.,
                       'cost':float(g.holding_cost.sum()+g.shortage_cost.sum())})
class_summary=pd.DataFrame(class_rows)
tight=class_summary.loc[(class_summary.capacity_factor==.75)&class_summary.policy.isin(['calibrated_buffer','calibrated_lp'])]
display(tight.round(3))
pivot=tight.pivot(index='abc',columns='policy',values='fill_rate').rename(columns=LABELS)
fig,ax=plt.subplots(figsize=(7,4));pivot.plot.bar(ax=ax,color=['#2563eb','#059669'])
ax.set(title='Tight capacity: class-level service changes under LP priorities',ylabel='Unit fill rate',xlabel='ABC class',ylim=(0,1.05))
ax.tick_params(axis='x',rotation=0);ax.legend(fontsize=8);fig.tight_layout();show(fig,'07_class_priority_tradeoff')
class_summary.to_csv(ROOT/'reports/class_service_summary.csv',index=False)''')
code('''trace=records.loc[(records.capacity_factor==1.)&(records['product']==example)]
fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
observed=trace.loc[trace.policy=='calibrated_lp'].sort_values('week')
axes[0].bar(observed.week,observed.demand,color='#cbd5e1',label='Recorded demand proxy')
for policy in ['mean_point','normal_buffer','calibrated_lp']:
    g=trace.loc[trace.policy==policy].sort_values('week')
    axes[0].plot(g.week,g.ending_on_hand,'o-',color=COLORS[policy],label=LABELS[policy]+' ending stock')
axes[0].set(title=f'Product {example}: the same demand stream under different stock policies',ylabel='Units')
axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
axes[1].step(observed.week,observed.ordered,where='mid',label='Order placed',color='#2563eb')
axes[1].step(observed.week,observed.arrivals,where='mid',label='Order received two weeks later',color='#059669')
axes[1].set(ylabel='Units',xlabel='Review week');axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
fig.tight_layout();show(fig,'08_stock_and_order_flow')
# Each receipt equals an order from exactly two weeks earlier.
for _,g in records.groupby(['capacity_factor','policy','product']):
    g=g.sort_values('week')
    np.testing.assert_array_equal(g.arrivals.to_numpy()[:2],[0,0])
    np.testing.assert_array_equal(g.arrivals.to_numpy()[2:],g.ordered.to_numpy()[:-2])
weekly.to_csv(ROOT/'reports/weekly_capacity_audit.csv',index=False)
print('Two-week receipt timing verified for every simulated product and policy.')''')
md(r'''## 7. Report the findings with their limits

The results below are calculated from saved experiment tables. The LP's lower expected surrogate objective does not guarantee the lowest realized inventory cost. Its value also depends on capacity and assumed class penalties.''')
code('''tight_summary=summary.loc[summary.capacity_factor==.75].set_index('policy')
base=tight_summary.loc['calibrated_buffer']; lp=tight_summary.loc['calibrated_lp']
print(f'Tight capacity: LP cost reduction versus empirical-buffer proportional allocation = {100*(1-lp.total_cost/base.total_cost):.2f}%')
print(f'Tight capacity aggregate fill rate: proportional {base.fill_rate:.3%}; LP {lp.fill_rate:.3%}')
normal=summary.loc[summary.capacity_factor==1.].set_index('policy')
print(f'Normal capacity: empirical buffer fill rate {normal.loc["calibrated_buffer","fill_rate"]:.3%}; cost {normal.loc["calibrated_buffer","total_cost"]:.0f}')
print(f'Normal capacity: boost-only fill rate {normal.loc["boost_point","fill_rate"]:.3%}; cost {normal.loc["boost_point","total_cost"]:.0f}')
print('Saved 13 result tables, configuration, reusable model bundle and eight figures.')''')
md(r'''### Limits and extensions

- There are only 52 observed weeks and 13 simulated holdout weeks. Overlapping forecast targets and common product shocks create dependence. No long-run performance or nominal coverage guarantee is inferred.
- Observed purchases are a proxy, not identified latent demand. Inventory availability and true unfulfilled orders are absent from the source. Simulated outcomes are hypothetical, not observed retailer savings.
- ABC penalties and capacities are assumed. The cohort is class-balanced and selected from the initial training prefix; it is not a sales-weighted representation of all products.
- Interval coverage is below 90% for some classes. A more reliable uncertainty system would need additional chronological calibration windows, conditional calibration, and monitoring over a longer dataset.
- The LP is a myopic surrogate; integer rounding is feasible and audited, without a universal optimality claim. It excludes setup costs, procurement costs, variable lead times, shelf constraints, perishability and multistage demand/stockout timing.
- The common starting stock and known terminal horizon influence the comparison. Longer runs with warm-up, multiple cohorts and uncensored demand observations would strengthen it.

### References

- Tan, J. (2014), [Sales Transactions Weekly, UCI](https://doi.org/10.24432/C5XS4Q).
- MIT OpenCourseWare, [Stochastic-demand Inventory Models](https://ocw.mit.edu/courses/15-772j-d-lab-supply-chains-fall-2014/0d50c5c77382852102ee30b98f1d4657_MIT15_772JF14_Lec14.pdf).
- Angelopoulos and Bates, [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511). The standard exchangeability guarantee is not assumed here.
- SciPy, [linprog and HiGHS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).

### Run outside Jupyter

`python scripts/run_analysis.py` regenerates analytical tables and the fitted bundle. `python scripts/score_origin.py --origin 39` produces forecasts and bounds from that bundle. `python -m pytest -q` runs eleven tests of forecast availability, calibration and inventory accounting. See the README for environment setup and `docs/publishing.md` for GitHub publishing.''')
nb=n.v4.new_notebook(cells=c,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
n.write(nb,r/'notebooks/demand_inventory_optimization.ipynb');print(len(c),'cells',sum(x.cell_type=='code' for x in c),'code cells')
