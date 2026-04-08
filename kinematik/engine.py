
from dataclasses import dataclass

import numpy as np

from .geometry import PlatformGeometry
from .kinematik_main import func_kinematik_main
from .numba_abstande import NumbaAbstande
from .solver import KinematikSolver


# Servowinkel in Grad; actual_angle kann von requested abweichen wenn Erreichbarkeit korrigiert hat
@dataclass
class KinematicsResult:
    servo_v: float
    servo_l: float
    servo_r: float
    actual_angle_x: float
    actual_angle_y: float
    solver_success: bool
    reachability_ok: bool
    n_vek: np.ndarray
    d: float


class KinematicsEngine:
    # einmal beim Start instanziieren, dann solve() im Regelloop aufrufen
    def __init__(
        self,
        geometry: PlatformGeometry,
        solver_method: str = "scipy",
        solver_tol: float = 1e-3,
        solver_max_iter: int = 50,
    ) -> None:
        self.geometry = geometry

        # Dummy-Richtungen, werden bei jedem solve() überschrieben
        s_r = np.array([1.0, -0.4, 0.9], dtype=np.float64)
        s_v = np.array([0.8, 0.6, -0.7], dtype=np.float64)
        s_l = np.array([-0.5, 0.3, 1.1], dtype=np.float64)

        abstande = NumbaAbstande(s_r=s_r, s_l=s_l, s_v=s_v, l=geometry.platform_link_length)
        abstande.warmup()

        self._solver = KinematikSolver(
            abstande=abstande,
            method=solver_method,
            tol=solver_tol,
            max_iter=solver_max_iter,
        )

    def solve(self, winkel_x: float, winkel_y: float, ref_point: np.ndarray | None = None) -> KinematicsResult:
        if ref_point is None:
            ref_point = self.geometry.ref_point

        g = self.geometry
        raw = func_kinematik_main(
            g.arm_pos_front, g.arm_pos_left, g.arm_pos_right,
            ref_point, winkel_x, winkel_y,
            g.upper_arm_length, g.lower_arm_length,
            self._solver,
        )

        return KinematicsResult(
            servo_v=float(raw["phi_servo_v"]),
            servo_l=float(raw["phi_servo_l"]),
            servo_r=float(raw["phi_servo_r"]),
            actual_angle_x=float(raw["winkel_x"]),
            actual_angle_y=float(raw["winkel_y"]),
            solver_success=bool(raw.get("solver_success", True)),
            reachability_ok=bool(raw.get("reachability_ok", True)),
            n_vek=np.asarray(raw["n_vek"], dtype=np.float64),
            d=float(raw["d"]),
        )
