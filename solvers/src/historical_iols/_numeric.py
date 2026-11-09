from __future__ import annotations

from typing import Any

import numpy as np

FLOAT64_MAX = float(np.finfo(np.float64).max)
FLOAT64_LOG_MAX = float(np.log(FLOAT64_MAX))


def exp_with_float64_overflow_guard(values: Any) -> np.ndarray:
    """Exponentiate in float64, saturating only above the representable boundary."""

    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array, dtype=np.float64)
    overflow_mask = np.isposinf(array) | (array > FLOAT64_LOG_MAX)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        np.exp(array, out=result, where=~overflow_mask)

    if np.any(overflow_mask):
        result[overflow_mask] = FLOAT64_MAX
    return result
