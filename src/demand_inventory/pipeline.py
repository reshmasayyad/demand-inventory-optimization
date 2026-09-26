"""One reproducible workflow shared by the CLI and notebook."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from .data import load_sales, product_metadata, select_inventory_cohort
from .forecasting import select_complexity, fit_forecaster, baseline_prediction, forecast_records, forecast_scores
from .uncertainty import calibrate, coverage_report
from .inventory import POLICIES, simulate, summarize_inventory


def run(root, save=True):
    root = Path(root)
    codes, values = load_sales(root / 'data/raw/Sales_Transactions_Dataset_Weekly.csv')
    initial_metadata = product_metadata(codes, values, cutoff=24)
    cohort = select_inventory_cohort(initial_metadata, each_class=10, seed=42)
    leaves, selection_scores = select_complexity(values, initial_metadata)
    classes = initial_metadata.set_index('product').abc
    metadata = product_metadata(codes, values, cutoff=30, classes=classes)
    model = fit_forecaster(values, metadata, cutoff=30, leaves=leaves)
    calibration_records = forecast_records(values, metadata, range(30, 37), {'boosting': model.predict})
    calibration = calibrate(calibration_records, coverage=.90)
    predictors = {name: (lambda v, t, name=name: baseline_prediction(v, t, name)) for name in ['last_week', 'mean4', 'mean13']}
    predictors['boosting'] = model.predict
    test_records = forecast_records(values, metadata, range(39, 50), predictors)
    test_scores = forecast_scores(test_records)
    coverage, intervals = coverage_report(test_records.loc[test_records.model == 'boosting'], calibration)
    inventory, audits = [], []
    for capacity_factor in [.75, 1.0, 1.25]:
        for policy in POLICIES:
            records, audit = simulate(values, metadata, model, calibration, cohort, policy, capacity_factor)
            inventory.append(records)
            if not audit.empty:
                audit['capacity_factor'] = capacity_factor; audits.append(audit)
    inventory_records = pd.concat(inventory, ignore_index=True)
    inventory_summary = summarize_inventory(inventory_records)
    allocation_audit = pd.concat(audits, ignore_index=True)
    config = {'products': len(codes), 'weeks': values.shape[1], 'horizon': 3,
              'initial_fit_cutoff': 24, 'selection_origins': list(range(24, 28)),
              'refit_cutoff': 30, 'calibration_origins': list(range(30, 37)),
              'last_calibration_label': calibration.last_label_week,
              'test_forecast_origins': list(range(39, 50)), 'simulation_weeks': list(range(39, 52)),
              'selected_max_leaf_nodes': leaves, 'nominal_coverage': .90,
              'inventory_cohort_products': metadata.iloc[cohort]['product'].tolist(), 'cohort_seed': 42,
              'lead_weeks': 2, 'review_weeks': 1, 'LP_scenarios': 20,
              'capacity_factors': [.75, 1.0, 1.25], 'holding_cost': 1.0,
              'shortage_penalties': {'A': 20, 'B': 10, 'C': 5}}
    tables = {'product_metadata': metadata, 'inventory_cohort': metadata.iloc[cohort].reset_index(drop=True),
              'selection_scores': selection_scores, 'test_forecast_scores': test_scores,
              'calibration_predictions': calibration_records, 'test_predictions': test_records,
              'interval_coverage': coverage, 'test_intervals': intervals,
              'inventory_records': inventory_records, 'inventory_summary': inventory_summary,
              'allocation_audit': allocation_audit}
    if save:
        reports = root / 'reports'; reports.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(reports / f'{name}.csv', index=False)
        (reports / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
        joblib.dump({'model': model, 'calibration': calibration, 'cohort': cohort}, reports / 'forecast_bundle.joblib')
    return {**tables, 'codes': codes, 'values': values, 'initial_metadata': initial_metadata,
            'model': model, 'calibration': calibration, 'cohort': cohort, 'configuration': config}
