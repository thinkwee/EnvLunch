"""Canonical form and agreement of dataframe results (scalars, mappings, frames)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MAX_FRAME_CELLS = 2000
ROUND = 4


def _scalar(v):
    """Finite float or None."""
    try:
        f = float(v)
    except (OverflowError, ValueError, TypeError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def normalize(val):
    """Canonicalise a result for comparison."""
    if val is None:
        return None
    if isinstance(val, (int, float, np.integer, np.floating)):
        return _scalar(val)
    if isinstance(val, np.ndarray) and val.size == 1:
        return normalize(val.item())
    if isinstance(val, pd.DataFrame):
        if val.size == 0 or val.size > MAX_FRAME_CELLS:
            return None
        rows = []
        for _, row in val.iterrows():
            cells = []
            for v in row:
                if isinstance(v, (int, float, np.integer, np.floating)):
                    f = _scalar(v)
                    cells.append(None if f is None else round(f, ROUND))
                else:
                    cells.append(str(v))
            rows.append(tuple(cells))
        return ("frame", tuple(str(c) for c in val.columns), tuple(sorted(rows, key=repr)))
    if isinstance(val, pd.Series):
        val = val.to_dict()
    if isinstance(val, dict):
        if not 0 < len(val) <= 50:
            return None
        out = []
        for k, v in val.items():
            f = normalize(v)
            if not isinstance(f, float):
                return None
            out.append((str(k), f))
        return tuple(sorted(out))
    return None


def _close(a, b) -> bool:
    """Equality with numeric tolerance."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-4)
    return a == b


def _is_frame(x) -> bool:
    """Whether a canonical result is a frame."""
    return isinstance(x, tuple) and len(x) == 3 and x[0] == "frame"


def _is_mapping(x) -> bool:
    """Whether a canonical result is a mapping."""
    return isinstance(x, tuple) and not _is_frame(x) and bool(x) and all(
        isinstance(e, tuple) and len(e) == 2 for e in x)


def match(a, b) -> float:
    """Agreement m of two canonical results."""
    if a is None or b is None:
        return 0.0
    if _is_frame(a) or _is_frame(b):
        if not (_is_frame(a) and _is_frame(b)) or a[1] != b[1]:
            return 0.0
        sa, sb = set(a[2]), set(b[2])
        return 1.0 if not sa and not sb else len(sa & sb) / max(len(sa), len(sb))
    if _is_mapping(a) or _is_mapping(b):
        if not (_is_mapping(a) and _is_mapping(b)):
            return 0.0
        db = dict(b)
        hit = sum(1 for k, v in a if k in db and _close(v, db[k]))
        return hit / max(len(a), len(b))
    return 1.0 if _close(a, b) else 0.0
