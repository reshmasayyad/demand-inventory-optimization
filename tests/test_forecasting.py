import numpy as np
import pandas as pd
import pytest
from demand_inventory.data import load_sales, product_metadata, WEEKS
from demand_inventory.features import features_at, training_examples
from demand_inventory.uncertainty import rank_quantile, calibrate


def test_raw_loader_ignores_full_history_normalization_and_rejects_bad_counts(tmp_path):
    frame = pd.DataFrame({'Product_Code': ['P1', 'P2'], **{w: [1, 2] for w in WEEKS}, 'MAX': [9999, 9999], 'Normalized 51': [100, -100]})
    path = tmp_path / 'sales.csv'; frame.to_csv(path, index=False)
    codes, values = load_sales(path)
    np.testing.assert_array_equal(values, np.tile([[1], [2]], (1, 52)))
    frame['W51'] = frame['W51'].astype(float)
    frame.loc[0, 'W51'] = -.5; frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='nonnegative integers'):
        load_sales(path)


def test_classification_and_scaling_exclude_future_observations():
    values = np.vstack([np.full(52, 30), np.full(52, 10), np.full(52, 1)]).astype(float)
    codes = ['A', 'B', 'C']
    first = product_metadata(codes, values, cutoff=24)
    changed = values.copy(); changed[:, 24:] = [[0], [100000], [5000]]
    second = product_metadata(codes, changed, cutoff=24)
    pd.testing.assert_frame_equal(first, second)


def test_current_and_future_values_cannot_change_origin_features():
    rng = np.random.default_rng(42); values = rng.poisson(10, size=(3, 52)).astype(float)
    meta = product_metadata(['a', 'b', 'c'], values, cutoff=30)
    first = features_at(values, meta, 39)
    changed = values.copy(); changed[:, 39:] = 100000
    pd.testing.assert_frame_equal(first, features_at(changed, meta, 39))
    changed[:, 38] += 10
    assert not first.equals(features_at(changed, meta, 39))


def test_training_examples_require_all_labels_before_fit_cutoff():
    values = np.tile(np.arange(52), (2, 1)).astype(float)
    meta = product_metadata(['a', 'b'], values, cutoff=30)
    x, y = training_examples(values, meta, cutoff=30)
    assert len(x) == 2 * 15
    np.testing.assert_allclose(y[-2:], np.array([27+28+29]*2) / meta.scale)
    changed = values.copy(); changed[:, 30:] = 100000
    xx, yy = training_examples(changed, meta, cutoff=30)
    pd.testing.assert_frame_equal(x, xx); np.testing.assert_array_equal(y, yy)


def test_finite_sample_quantile_handles_insufficient_scores_explicitly():
    assert rank_quantile(range(9), .8) == 7
    assert np.isinf(rank_quantile([1], .9))
    with pytest.raises(ValueError): rank_quantile([np.nan], .9)


def test_calibration_records_last_target_label_and_nonnegative_buffers():
    f = pd.DataFrame({'model': ['boosting']*30, 'abc': ['A']*30, 'origin': [36]*30,
                      'actual': [1]*30, 'prediction': [10]*30, 'scale': [1]*30})
    result = calibrate(f)
    assert result.last_label_week == 38
    assert result.upper_buffer['A'] == 0
    assert result.radius['A'] == 9
