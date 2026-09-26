"""Generate protection-period forecasts and bounds from the saved bundle."""
import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from demand_inventory.data import load_sales
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/raw/Sales_Transactions_Dataset_Weekly.csv')
    parser.add_argument('--origin', type=int, default=39)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/forecast_targets.csv')
    args = parser.parse_args()
    package = joblib.load(ROOT / 'reports/forecast_bundle.joblib')
    codes, values = load_sales(args.data)
    model, calibration = package['model'], package['calibration']
    if codes.tolist() != model.metadata['product'].tolist():
        raise ValueError('Input product order/IDs must match the fitted metadata')
    if args.origin <= calibration.last_label_week:
        raise ValueError('Origin must follow calibration label availability')
    point = model.predict(values, args.origin)
    lower, upper, target = calibration.bounds(point, model.metadata)
    output = pd.DataFrame({'product': codes, 'origin': args.origin, 'three_week_forecast': point,
                           'lower_90': lower, 'upper_90': upper, 'upper_order_target_90': target})
    args.output.parent.mkdir(parents=True, exist_ok=True); output.to_csv(args.output, index=False)
    print(f'Wrote {len(output)} product forecasts to {args.output}')
