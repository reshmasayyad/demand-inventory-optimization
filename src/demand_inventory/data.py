"""Read raw weekly quantities; ignore full-history normalization columns."""
from pathlib import Path
import numpy as np
import pandas as pd

WEEKS = [f'W{i}' for i in range(52)]


def load_sales(path: str | Path):
    raw = pd.read_csv(path)
    if not {'Product_Code', *WEEKS}.issubset(raw.columns):
        raise ValueError('Expected product IDs and W0 through W51')
    if raw.Product_Code.duplicated().any():
        raise ValueError('Duplicate product IDs')
    values = raw[WEEKS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Weekly quantities must be finite nonnegative integers')
    return raw.Product_Code.astype(str).to_numpy(), values


def product_metadata(codes, values, cutoff=24, classes=None):
    past = values[:, :cutoff]
    total = past.sum(axis=1)
    metadata = pd.DataFrame({'product': codes, 'training_total': total, 'mean': past.mean(axis=1),
                             'scale': np.maximum(past.mean(axis=1), 1.0),
                             'zero_fraction': (past == 0).mean(axis=1), 'std': past.std(axis=1, ddof=1)})
    if classes is None:
        ranked = metadata.sort_values(['training_total', 'product'], ascending=[False, True]).copy()
        denominator = max(float(ranked.training_total.sum()), 1.0)
        share_before = (ranked.training_total.cumsum() - ranked.training_total) / denominator
        ranked['abc'] = np.select([share_before < .80, share_before < .95], ['A', 'B'], default='C')
        metadata['abc'] = metadata['product'].map(ranked.set_index('product').abc)
    else:
        metadata['abc'] = metadata['product'].map(classes)
        if metadata.abc.isna().any():
            raise ValueError('Class mapping does not cover every product')
    return metadata


def select_inventory_cohort(metadata, each_class=10, seed=42):
    """Class-balanced random sample from products active before the selection cutoff."""
    rng = np.random.default_rng(seed)
    selected = []
    for group in ['A', 'B', 'C']:
        candidates = metadata.loc[(metadata.abc == group) & (metadata.training_total > 0)].index.to_numpy()
        if len(candidates) < each_class:
            raise ValueError(f'Insufficient active products in class {group}')
        selected.extend(rng.choice(candidates, each_class, replace=False).tolist())
    return np.array(sorted(selected), dtype=int)
