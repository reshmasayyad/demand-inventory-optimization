"""Origin-indexed features that use observations strictly before a review week."""
import numpy as np
import pandas as pd
HORIZON = 3
FEATURES = ['lag1', 'lag2', 'lag4', 'mean4', 'mean8', 'mean13', 'std13', 'zero13', 'recent_trend', 'log_scale', 'abc_code']


def features_at(values, metadata, origin):
    if origin < 13 or origin > values.shape[1]:
        raise ValueError('Need thirteen observed weeks before the origin')
    if len(metadata) != len(values):
        raise ValueError('Metadata must align with product rows')
    past = values[:, origin - 13:origin]
    scale = metadata.scale.to_numpy()
    mean4, mean13 = past[:, -4:].mean(axis=1), past.mean(axis=1)
    return pd.DataFrame({
        'lag1': past[:, -1] / scale, 'lag2': past[:, -2] / scale, 'lag4': past[:, -4] / scale,
        'mean4': mean4 / scale, 'mean8': past[:, -8:].mean(axis=1) / scale, 'mean13': mean13 / scale,
        'std13': past.std(axis=1, ddof=1) / scale, 'zero13': (past == 0).mean(axis=1),
        'recent_trend': (mean4 - mean13) / scale, 'log_scale': np.log1p(scale),
        'abc_code': metadata.abc.map({'A': 0, 'B': 1, 'C': 2}).to_numpy(),
    })[FEATURES]


def training_examples(values, metadata, cutoff, horizon=HORIZON):
    if cutoff > values.shape[1] or cutoff < 13 + horizon:
        raise ValueError('Invalid training cutoff')
    xs, ys = [], []
    for origin in range(13, cutoff - horizon + 1):
        xs.append(features_at(values, metadata, origin))
        ys.append(values[:, origin:origin + horizon].sum(axis=1) / metadata.scale.to_numpy())
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)
