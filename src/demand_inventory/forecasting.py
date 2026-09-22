"""Pooled forecasts and rolling-origin comparisons of three-week totals."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .features import FEATURES, HORIZON, features_at, training_examples


@dataclass
class ForecastModel:
    estimator: object
    metadata: pd.DataFrame
    fit_cutoff: int
    leaves: int

    def predict(self, values, origin):
        if origin < self.fit_cutoff:
            raise ValueError('Prediction origin predates the fitted model availability')
        normalized = self.estimator.predict(features_at(values, self.metadata, origin))
        return np.maximum(normalized * self.metadata.scale.to_numpy(), 0.0)


def fit_forecaster(values, metadata, cutoff, leaves):
    x, y = training_examples(values, metadata, cutoff)
    estimator = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=leaves, learning_rate=.06,
        l2_regularization=3, min_samples_leaf=30, random_state=42, early_stopping=False,
        categorical_features=[False] * (len(FEATURES) - 1) + [True])
    estimator.fit(x, y)
    return ForecastModel(estimator, metadata.copy(), cutoff, leaves)


def baseline_prediction(values, origin, name):
    if name == 'last_week':
        return HORIZON * values[:, origin - 1]
    if name not in {'mean4', 'mean13'}:
        raise ValueError('Unknown baseline')
    window = 4 if name == 'mean4' else 13
    return HORIZON * values[:, origin - window:origin].mean(axis=1)


def forecast_records(values, metadata, origins, predictors):
    rows = []
    for origin in origins:
        if origin + HORIZON > values.shape[1]:
            raise ValueError('Evaluation target extends beyond available data')
        actual = values[:, origin:origin + HORIZON].sum(axis=1)
        for name, predictor in predictors.items():
            predicted = predictor(values, origin)
            rows.append(pd.DataFrame({'product': metadata['product'], 'abc': metadata.abc, 'origin': origin,
                                      'actual': actual, 'prediction': predicted, 'scale': metadata.scale, 'model': name}))
    return pd.concat(rows, ignore_index=True)


def forecast_scores(records):
    rows = []
    for name, group in records.groupby('model', sort=False):
        error = group.actual - group.prediction
        denominator = group.actual.sum()
        rows.append({'model': name, 'mae': np.abs(error).mean(), 'rmse': np.sqrt(np.mean(error**2)),
                     'wape': np.abs(error).sum() / denominator if denominator else np.nan,
                     'normalized_mae': (np.abs(error) / group.scale).mean(), 'predictions': len(group)})
    return pd.DataFrame(rows).sort_values('mae').reset_index(drop=True)


def select_complexity(values, metadata):
    predictors = {name: (lambda v, t, name=name: baseline_prediction(v, t, name)) for name in ['last_week', 'mean4', 'mean13']}
    for leaves in [15, 31]:
        fitted = fit_forecaster(values, metadata, cutoff=24, leaves=leaves)
        predictors[f'boosting_{leaves}'] = fitted.predict
    records = forecast_records(values, metadata, range(24, 28), predictors)
    scores = forecast_scores(records)
    # Baselines are reported; complexity selection is restricted to the boosting family.
    selected = scores.loc[scores.model.str.startswith('boosting_')].iloc[0].model
    return int(selected.split('_')[1]), scores
