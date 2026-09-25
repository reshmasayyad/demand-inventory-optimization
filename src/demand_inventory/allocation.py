"""Scenario LP for myopic allocation of a common unit-capacity budget."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def objective(demands, position, orders, penalties, holding=1.0):
    net = position[:, None] + orders[:, None] - demands
    return float(np.mean(holding * np.maximum(net, 0) + penalties[:, None] * np.maximum(-net, 0), axis=1).sum())


def allocate_scenarios(demands, position, requests, capacity, penalties, holding=1.0):
    demands = np.asarray(demands, dtype=float)
    position, requests, penalties = [np.asarray(x, dtype=float) for x in (position, requests, penalties)]
    n, scenarios = demands.shape
    if n == 0 or scenarios == 0 or len(position) != n or len(requests) != n or len(penalties) != n:
        raise ValueError('Incompatible allocation arrays')
    if capacity < 0 or holding <= 0 or (demands < 0).any() or (requests < 0).any() or (penalties <= 0).any():
        raise ValueError('Invalid capacity, requests, demand or costs')
    k = n * scenarios
    rows = np.tile(np.arange(k), 3)
    cols = np.r_[np.repeat(np.arange(n), scenarios), n + np.arange(k), n + k + np.arange(k)]
    coefficients = np.r_[np.ones(k), np.ones(k), -np.ones(k)]
    equality = coo_matrix((coefficients, (rows, cols)), shape=(k, n + 2*k)).tocsr()
    rhs = (demands - position[:, None]).ravel()
    capacity_row = coo_matrix((np.ones(n), (np.zeros(n, dtype=int), np.arange(n))), shape=(1, n + 2*k)).tocsr()
    costs = np.r_[np.zeros(n), np.repeat(penalties / scenarios, scenarios), np.full(k, holding / scenarios)]
    bounds = [(0, float(x)) for x in requests] + [(0, None)] * (2*k)
    solution = linprog(costs, A_ub=capacity_row, b_ub=[capacity], A_eq=equality, b_eq=rhs,
                       bounds=bounds, method='highs')
    if not solution.success:
        raise RuntimeError(f'Allocation LP failed: {solution.message}')
    relaxed = np.clip(solution.x[:n], 0, requests)
    allocated = np.floor(relaxed + 1e-7).astype(int)
    # Feasible integer correction: add units while they reduce the scenario objective.
    remaining = int(capacity - allocated.sum())
    while remaining > 0:
        available = allocated < np.floor(requests).astype(int)
        if not available.any():
            break
        net = position[:, None] + allocated[:, None] - demands
        now = holding * np.maximum(net, 0) + penalties[:, None] * np.maximum(-net, 0)
        later = holding * np.maximum(net + 1, 0) + penalties[:, None] * np.maximum(-(net + 1), 0)
        marginal = np.mean(later - now, axis=1)
        marginal[~available] = np.inf
        index = int(np.argmin(marginal))
        if marginal[index] >= -1e-9:
            break
        allocated[index] += 1; remaining -= 1
    integer_objective = objective(demands, position, allocated, penalties, holding)
    assert allocated.sum() <= capacity and (allocated <= requests).all() and (allocated >= 0).all()
    return allocated, {'lp_objective': float(solution.fun), 'integer_objective': integer_objective,
                       'rounding_gap': max(integer_objective - float(solution.fun), 0.0),
                       'used_capacity': int(allocated.sum()), 'requested_units': int(requests.sum())}
