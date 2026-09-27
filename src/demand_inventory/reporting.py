"""Reusable report figures; notebook execution displays their saved PNGs."""
import numpy as np
import matplotlib.pyplot as plt

LABELS = {'mean_point': 'Mean only', 'normal_buffer': 'Normal buffer', 'boost_point': 'Boost only',
          'calibrated_buffer': 'Empirical buffer', 'calibrated_lp': 'Buffer + LP'}
COLORS = dict(zip(LABELS, ['#64748b', '#f59e0b', '#7c3aed', '#2563eb', '#059669']))


def pareto(metadata):
    ranked = metadata.sort_values('training_total', ascending=False)
    cumulative = ranked.training_total.cumsum() / ranked.training_total.sum()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(np.arange(1, len(ranked)+1), cumulative, color='#2563eb')
    for y in [.8, .95]: ax.axhline(y, color='#94a3b8', linestyle='--')
    ax.set(xlabel='Products ranked by W0–W23 volume', ylabel='Cumulative share of recorded units',
           title='ABC classes use only the initial training period', ylim=(0, 1.02))
    fig.tight_layout(); return fig


def forecast_comparison(scores):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(scores.model, scores.mae, color=['#059669' if name == 'mean13' else '#64748b' for name in scores.model])
    ax.set(ylabel='MAE of three-week total (units)', title='Forecast accuracy: a simple baseline is competitive')
    fig.tight_layout(); return fig


def coverage_plot(coverage):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(coverage)); width=.32
    ax.bar(x-width/2, coverage.interval_coverage, width, label='Two-sided 90% interval', color='#2563eb')
    ax.bar(x+width/2, coverage.upper_target_coverage, width, label='One-sided 90% upper target', color='#059669')
    ax.axhline(.9, linestyle='--', color='#dc2626', label='Nominal level')
    ax.set_xticks(x, coverage.abc); ax.set(ylabel='Observed test coverage', xlabel='Frozen ABC class', ylim=(0, 1.08),
                                         title='Empirical test coverage varies across product classes')
    ax.legend(fontsize=8, loc='lower left'); fig.tight_layout(); return fig


def cost_panels(summary):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0), sharey=True)
    for ax, (factor, group) in zip(axes, summary.groupby('capacity_factor', sort=True)):
        group=group.set_index('policy').reindex(list(LABELS)); x=np.arange(len(group))
        ax.bar(x, group.holding_cost, color='#93c5fd', label='Holding')
        ax.bar(x, group.shortage_cost, bottom=group.holding_cost, color='#fb7185', label='Weighted lost sales')
        ax.set_xticks(x, [LABELS[name] for name in group.index], rotation=35, ha='right')
        ax.set_title(f'Capacity = {factor:.2f} × training mean')
    axes[0].set_ylabel('Simulated relevant cost units, W39–W51')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.94), ncol=2, fontsize=8)
    fig.suptitle('Inventory costs depend on ordering rules and available supply', fontsize=12, y=.99)
    fig.tight_layout(rect=(0,0,1,.84)); return fig


def service_cost_panels(summary):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for ax, (factor, group) in zip(axes, summary.groupby('capacity_factor', sort=True)):
        for row in group.itertuples():
            ax.scatter(row.total_cost, row.fill_rate, color=COLORS[row.policy], label=LABELS[row.policy], s=65)
        ax.set(title=f'Capacity factor {factor:.2f}', xlabel='Holding + weighted lost-sales cost')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Aggregate unit fill rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8)
    fig.suptitle('Cost and service show different tradeoffs'); fig.tight_layout(rect=(0, .10, 1, .96)); return fig
