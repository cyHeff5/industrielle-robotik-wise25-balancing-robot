from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=False, fastmath=True)
def _func_abstande_kernel(a: np.ndarray, s_r: np.ndarray, s_v: np.ndarray, s_l: np.ndarray, l: float) -> np.ndarray:
    tv = a[0]
    tl = a[1]
    tr = a[2]
    l2 = l * l

    f0 = (s_l[0] * tl) ** 2 + (s_v[1] * tv - s_l[1] * tl) ** 2 + (s_v[2] * tv - s_l[2] * tl) ** 2 - l2
    f1 = (s_v[0] * tv - s_r[0] * tr) ** 2 + (s_v[1] * tv - s_r[1] * tr) ** 2 + (s_v[2] * tv - s_r[2] * tr) ** 2 - l2
    f2 = (s_l[0] * tl - s_r[0] * tr) ** 2 + (s_l[1] * tl - s_r[1] * tr) ** 2 + (s_l[2] * tl - s_r[2] * tr) ** 2 - l2

    out = np.empty(3, dtype=np.float64)
    out[0] = f0
    out[1] = f1
    out[2] = f2
    return out


class NumbaAbstande:
    """Numba-accelerated implementation of func_abstande."""

    def __init__(self, s_r: np.ndarray, s_v: np.ndarray, s_l: np.ndarray, l: float = 216.5064) -> None:
        self.s_r = np.asarray(s_r, dtype=np.float64)
        self.s_v = np.asarray(s_v, dtype=np.float64)
        self.s_l = np.asarray(s_l, dtype=np.float64)
        self.l = float(l)

    def warmup(self) -> None:
        a0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        _func_abstande_kernel(a0, self.s_r, self.s_v, self.s_l, self.l)

    def evaluate(self, a: np.ndarray) -> np.ndarray:
        a_arr = np.asarray(a, dtype=np.float64)
        return _func_abstande_kernel(a_arr, self.s_r, self.s_v, self.s_l, self.l)
