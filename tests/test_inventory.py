import numpy as np
import pandas as pd
import pytest
from demand_inventory.inventory import simulate, proportional_allocate
from demand_inventory.uncertainty import ResidualCalibration
from demand_inventory.allocation import allocate_scenarios, objective


class PastMeanModel:
    def predict(self, values, origin):
        return 3 * values[:, origin-13:origin].mean(axis=1)


def case():
    values = np.full((3, 52), 10.0)
    meta = pd.DataFrame({'product': ['a', 'b', 'c'], 'abc': ['A', 'B', 'C'], 'mean': [10]*3, 'scale': [10]*3})
    zero = {g: 0 for g in 'ABC'}
    cal = ResidualCalibration({g: np.zeros(20) for g in 'ABC'}, zero, zero, .90, 38)
    return values, meta, PastMeanModel(), cal


def test_proportional_allocation_is_integer_and_respects_capacity():
    np.testing.assert_array_equal(proportional_allocate([5, 5, 5], 7), [3, 2, 2])
    np.testing.assert_array_equal(proportional_allocate([0, 0], 7), [0, 0])
    with pytest.raises(ValueError): proportional_allocate([-1, 2], 7)


def test_receipts_respect_two_week_lead_and_units_are_conserved():
    values, meta, model, cal = case()
    records, _ = simulate(values, meta, model, cal, [0, 1, 2], 'boost_point', start=39, end=45)
    for _, group in records.groupby('product'):
        ordered = group.ordered.to_numpy(); arriving = group.arrivals.to_numpy()
        np.testing.assert_array_equal(arriving[:2], [0, 0])
        np.testing.assert_array_equal(arriving[2:], ordered[:-2])
        previous = np.r_[20, group.ending_on_hand.to_numpy()[:-1]]
        np.testing.assert_array_equal(previous + arriving - group.fulfilled, group.ending_on_hand)
    assert records.lost.sum() == 0
    assert records.loc[records.week >= 43, 'ordered'].sum() == 0
    assert records.loc[records.week == 44, 'ending_pipeline'].sum() == 0


def test_same_week_demand_is_revealed_after_order_and_capacity_is_shared():
    values, meta, model, cal = case()
    base, _ = simulate(values, meta, model, cal, [0, 1, 2], 'boost_point', capacity_factor=.5)
    changed = values.copy(); changed[:, 39:] = 10000
    perturbed, _ = simulate(changed, meta, model, cal, [0, 1, 2], 'boost_point', capacity_factor=.5)
    np.testing.assert_array_equal(base.loc[base.week == 39, 'ordered'], perturbed.loc[perturbed.week == 39, 'ordered'])
    assert (base.groupby('week').ordered.sum() <= 15).all()
    assert (base.demand == base.fulfilled + base.lost).all()


def test_calibration_cannot_be_used_before_its_targets_arrive():
    values, meta, model, cal = case()
    with pytest.raises(ValueError, match='availability'):
        simulate(values, meta, model, cal, [0, 1, 2], 'boost_point', start=38)


def test_scenario_lp_matches_small_integer_enumeration_and_lower_bound():
    demand = np.array([[3., 5.], [2., 4.]])
    position, requests, penalties = np.array([1., 0.]), np.array([4., 4.]), np.array([10., 3.])
    orders, audit = allocate_scenarios(demand, position, requests, 3, penalties)
    best = min(objective(demand, position, np.array([a, b]), penalties)
               for a in range(5) for b in range(5) if a+b <= 3)
    assert objective(demand, position, orders, penalties) == pytest.approx(best)
    assert orders.sum() <= 3 and np.equal(orders, np.floor(orders)).all()
    assert audit['lp_objective'] <= audit['integer_objective'] + 1e-7
    zero, _ = allocate_scenarios(demand, position, requests, 0, penalties)
    np.testing.assert_array_equal(zero, [0, 0])
