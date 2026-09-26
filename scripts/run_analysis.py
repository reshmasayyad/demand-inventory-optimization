"""Run forecasting, calibration and inventory experiments without Jupyter."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from demand_inventory.pipeline import run
if __name__ == '__main__':
    result = run(ROOT)
    print('Forecast comparison:\n', result['test_forecast_scores'].round(3).to_string(index=False))
    print('\nInventory cases:\n', result['inventory_summary'].round(3).to_string(index=False))
    print('\nInterval coverage:\n', result['interval_coverage'].round(3).to_string(index=False))
