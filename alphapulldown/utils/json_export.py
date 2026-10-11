"""
JSON copies of derived prediction data (alongside the pickles, which stay as
they are): readable without unpickling, from any language, and independent of
numpy/Python versions. Written as .json.xz; floats rounded, since these are
probabilities and the full-precision L x L matrix would otherwise be several
times the size of the pickle.
"""
import json
import lzma

import numpy as np


def to_jsonable(obj, ndigits=3):
    """numpy arrays/scalars -> lists/Python numbers, floats rounded, tuples -> lists."""
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.floating):
            return np.round(obj, ndigits).tolist()
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), ndigits)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, ndigits) for v in obj]
    return obj


def write_json_xz(path, obj, ndigits=3):
    """Write obj as xz-compressed JSON; NaN/inf raise rather than produce invalid JSON."""
    with lzma.open(path, "wt") as ofile:
        json.dump(to_jsonable(obj, ndigits), ofile, allow_nan=False)
