"""Class-pooled, scale-normalized residual calibration and demand scenarios."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


def rank_quantile(scores, coverage):
    if not 0 < coverage < 1:
        raise ValueError('Coverage must be strictly between zero and one')
    scores = np.sort(np.asarray(scores, dtype=float))
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Need finite calibration scores')
    rank = int(np.ceil((len(scores) + 1) * coverage))
    return float(scores[rank - 1]) if rank <= len(scores) else float('inf')


@dataclass
class ResidualCalibration:
    signed: dict
    radius: dict
    upper_buffer: dict
    coverage: float
    last_label_week: int

    def bounds(self, point, metadata):
        scale = metadata.scale.to_numpy()
        radii = metadata.abc.map(self.radius).to_numpy() * scale
        buffers = metadata.abc.map(self.upper_buffer).to_numpy() * scale
        return np.maximum(point - radii, 0), point + radii, np.maximum(point + buffers, 0)

    def scenarios(self, point, metadata, count=20):
        levels = (np.arange(count) + .5) / count
        noise = np.stack([np.quantile(self.signed[group], levels) for group in metadata.abc])
        return np.maximum(point[:, None] + metadata.scale.to_numpy()[:, None] * noise, 0)


def calibrate(records, coverage=.90):
    if records.model.nunique() != 1:
        raise ValueError('Calibrate a single frozen forecast model')
    signed, radius, upper = {}, {}, {}
    for group, frame in records.groupby('abc'):
        scores = ((frame.actual - frame.prediction) / frame.scale).to_numpy()
        signed[group] = scores
        radius[group] = rank_quantile(np.abs(scores), coverage)
        upper[group] = max(rank_quantile(scores, coverage), 0.0)
    return ResidualCalibration(signed, radius, upper, coverage, int(records.origin.max() + 2))


def coverage_report(records, calibration):
    out = records.copy()
    point = out.prediction.to_numpy()
    lower, upper, target = calibration.bounds(point, out)
    out['lower'], out['upper'], out['upper_target'] = lower, upper, target
    out['covered'] = (out.actual >= lower) & (out.actual <= upper)
    out['upper_covered'] = out.actual <= target
    out['width'] = upper - lower
    rows = []
    for group, frame in out.groupby('abc'):
        rows.append({'abc': group, 'interval_coverage': frame.covered.mean(),
                     'upper_target_coverage': frame.upper_covered.mean(),
                     'mean_width': frame.width.mean(), 'predictions': len(frame)})
    return pd.DataFrame(rows), out
