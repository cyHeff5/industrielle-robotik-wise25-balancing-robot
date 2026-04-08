from __future__ import annotations

import numpy as np
from kinematik.numba_abstande import NumbaAbstande
from scipy.optimize import root

class KinematikSolver:
    def __init__(self, abstande: NumbaAbstande, method: str = "newton", tol: float = 1e-3, max_iter: int = 50, jac_eps: float = 1e-6, min_step: float = 1e-6, initial_guess: np.ndarray | None = None,) -> None:
        self.abstande = abstande
        self.method = method
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.jac_eps = float(jac_eps)
        self.min_step = float(min_step)
        self.last_solution = (
            np.asarray(initial_guess, dtype=np.float64).copy()
            if initial_guess is not None
            else np.array([100.0, 100.0, 100.0], dtype=np.float64)
        )

    def update_directions(self, s_v: np.ndarray, s_l: np.ndarray, s_r: np.ndarray) -> None:
        self.abstande.s_v[:] = np.asarray(s_v, dtype=np.float64)
        self.abstande.s_l[:] = np.asarray(s_l, dtype=np.float64)
        self.abstande.s_r[:] = np.asarray(s_r, dtype=np.float64)

    def residual(self, a: np.ndarray) -> np.ndarray:
        return self.abstande.evaluate(a)

    def solve(self, a0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, bool, int]:
        x0 = self.last_solution if a0 is None else np.asarray(a0, dtype=np.float64)
        if self.method == "scipy":
            return self._solve_scipy(x0)
        if self.method == "newton":
            return self._solve_newton(x0)
        raise ValueError(f"Unknown solver method '{self.method}'. Use 'newton' or 'scipy'.")

    def _jacobian_central(self, a: np.ndarray) -> np.ndarray:
        j = np.empty((3, 3), dtype=np.float64)
        for i in range(3):
            h = self.jac_eps * max(1.0, abs(a[i]))
            a_plus = a.copy()
            a_minus = a.copy()
            a_plus[i] += h
            a_minus[i] -= h
            f_plus = self.residual(a_plus)
            f_minus = self.residual(a_minus)
            j[:, i] = (f_plus - f_minus) / (2.0 * h)
        return j

    def _solve_newton(self, a0: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, int]:
        a = np.asarray(a0, dtype=np.float64).copy()
        f = self.residual(a)
        err = np.linalg.norm(f, ord=np.inf)

        if err < self.tol:
            self.last_solution = a.copy()
            return a, f, True, 0

        for it in range(1, self.max_iter + 1):
            j = self._jacobian_central(a)
            try:
                delta = np.linalg.solve(j, -f)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(j, -f, rcond=None)

            step = 1.0
            improved = False
            while step >= self.min_step:
                cand = a + step * delta
                f_cand = self.residual(cand)
                err_cand = np.linalg.norm(f_cand, ord=np.inf)
                if err_cand < err:
                    a = cand
                    f = f_cand
                    err = err_cand
                    improved = True
                    break
                step *= 0.5

            if not improved:
                return a, f, False, it

            if err < self.tol:
                self.last_solution = a.copy()
                return a, f, True, it

        return a, f, False, self.max_iter

    def _solve_scipy(self, a0: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, int]:
        res = root(lambda a: self.residual(a), a0, method="hybr", tol=self.tol)
        x = np.asarray(res.x, dtype=np.float64)
        f = np.asarray(res.fun, dtype=np.float64)
        ok = bool(res.success)
        n_iter = int(getattr(res, "nfev", 0))
        if ok:
            self.last_solution = x.copy()
        return x, f, ok, n_iter
