"""Weekly lost-sales simulation with receipts, ordering, demand and holding costs."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from .features import HORIZON

POLICIES = ['mean_point', 'normal_buffer', 'boost_point', 'calibrated_buffer', 'calibrated_lp']
PENALTIES = {'A': 20.0, 'B': 10.0, 'C': 5.0}


def proportional_allocate(requests, capacity):
    requests = np.asarray(requests, dtype=int)
    if (requests < 0).any() or capacity < 0:
        raise ValueError('Nonnegative requests and capacity required')
    if requests.sum() <= capacity:
        return requests.copy()
    raw = requests * (capacity / requests.sum())
    allocated = np.floor(raw).astype(int)
    remaining = int(capacity - allocated.sum())
    candidates = np.argsort(-(raw - allocated), kind='stable')
    for index in candidates:
        if remaining == 0:
            break
        if allocated[index] < requests[index]:
            allocated[index] += 1; remaining -= 1
    return allocated


def simulate(values, metadata, model, calibration, cohort, policy, capacity_factor=1.0,
             start=39, end=52, lead=2):
    if policy not in POLICIES:
        raise ValueError('Unknown inventory policy')
    if lead + 1 != HORIZON or start <= calibration.last_label_week:
        raise ValueError('Protection horizon or calibration availability mismatch')
    if not (0 <= start < end <= values.shape[1]) or capacity_factor <= 0:
        raise ValueError('Invalid simulation span or capacity')
    meta = metadata.iloc[cohort].reset_index(drop=True)
    selected = values[cohort]
    penalties = meta.abc.map(PENALTIES).to_numpy()
    capacity = int(np.ceil(capacity_factor * meta['mean'].sum()))
    # Common pre-test starting state, estimated from the observed refit period only.
    on_hand = np.ceil(lead * meta['mean'].to_numpy()).astype(int)
    pipeline, rows, optimization = {}, [], []
    for week in range(start, end):
        arriving = pipeline.pop(week, np.zeros(len(cohort), dtype=int))
        on_hand = on_hand + arriving
        position = on_hand + sum(pipeline.values(), start=np.zeros(len(cohort), dtype=int))
        point = model.predict(values, week)[cohort]
        past = selected[:, week - 13:week]
        if policy == 'mean_point':
            target = HORIZON * past.mean(axis=1)
        elif policy == 'normal_buffer':
            target = HORIZON * past.mean(axis=1) + norm.ppf(.90) * np.sqrt(HORIZON) * past.std(axis=1, ddof=1)
        elif policy == 'boost_point':
            target = point
        else:
            _, _, target = calibration.bounds(point, meta)
        requests = np.maximum(np.ceil(target).astype(int) - position, 0)
        # A finite-horizon comparison: no order is placed if it cannot arrive by W51.
        if week + lead >= end:
            orders = np.zeros(len(cohort), dtype=int)
            requests = np.zeros(len(cohort), dtype=int)
        elif policy == 'calibrated_lp':
            from .allocation import allocate_scenarios
            scenarios = calibration.scenarios(point, meta, count=20)
            orders, audit = allocate_scenarios(scenarios, position, requests, capacity, penalties)
            optimization.append({'week': week, **audit})
        else:
            orders = proportional_allocate(requests, capacity)
        assert orders.sum() <= capacity and (orders <= requests).all()
        pipeline[week + lead] = orders.copy()
        # Only now is this week's realized proxy demand exposed to the policy state.
        demand = selected[:, week].astype(int)
        fulfilled = np.minimum(on_hand, demand)
        lost = demand - fulfilled
        on_hand = on_hand - fulfilled
        assert (on_hand >= 0).all()
        pipeline_units = sum(pipeline.values(), start=np.zeros(len(cohort), dtype=int))
        rows.append(pd.DataFrame({'week': week, 'product': meta['product'], 'abc': meta.abc,
                                 'policy': policy, 'capacity_factor': capacity_factor, 'capacity': capacity,
                                 'inventory_position': position, 'arrivals': arriving, 'target': target,
                                 'requested': requests, 'ordered': orders, 'demand': demand,
                                 'fulfilled': fulfilled, 'lost': lost, 'ending_on_hand': on_hand,
                                 'ending_pipeline': pipeline_units, 'holding_cost': on_hand.astype(float),
                                 'shortage_cost': lost * penalties}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(optimization)


def summarize_inventory(records):
    rows = []
    for (factor, policy), group in records.groupby(['capacity_factor', 'policy'], sort=False):
        demand = group.demand.sum()
        last = group.loc[group.week == group.week.max()]
        rows.append({'capacity_factor': factor, 'policy': policy,
                     'total_cost': group.holding_cost.sum() + group.shortage_cost.sum(),
                     'holding_cost': group.holding_cost.sum(), 'shortage_cost': group.shortage_cost.sum(),
                     'fill_rate': group.fulfilled.sum() / demand if demand else 1.0,
                     'item_week_no_stockout_rate': (group.lost == 0).mean(),
                     'mean_on_hand_per_item_week': group.ending_on_hand.mean(),
                     'lost_units': int(group.lost.sum()), 'ordered_units': int(group.ordered.sum()),
                     'terminal_on_hand': int(last.ending_on_hand.sum()),
                     'terminal_pipeline': int(last.ending_pipeline.sum()),
                     'evaluated_item_weeks': len(group)})
    return pd.DataFrame(rows)
