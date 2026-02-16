import numpy as np
import math

from gpiozero import AngularServo

from kinematik.ebenen_gleichungen import *

from kinematik.numba_abstande import NumbaAbstande # für den Solver
from scipy.optimize import root

def func_kinematik_main(pos_arm_v, pos_arm_l, pos_arm_r, pos_ball, winkel_x, winkel_y, schwinge_o_l, schwinge_u_l, solver):

    ## Hauptablauf der Kinematik-Berechnung:

    winkel_x = math.radians(winkel_x)
    winkel_y = math.radians(winkel_y)

    #Rotationsmatrix berechnen:
    x_sin = math.sin(winkel_x)
    x_cos = math.cos(winkel_x)
    rx = np.array([[1, 0, 0], [0, x_cos, -x_sin], [0, x_sin, x_cos]]) # Rotationsmatrix um X

    y_sin = math.sin(winkel_y)
    y_cos = math.cos(winkel_y)
    ry = np.array([[y_cos, 0, y_sin], [0, 1, 0], [-y_sin, 0, y_cos]]) #Rotationsmatrix um Y

    rot_mat = rx @ ry

    # Noramelenvektor drehen:
    n_vek = rot_mat @ np.array([[0], [0], [1]])

    # Schnittgerade vorderer Link zur Ebene berechen
    a   = -n_vek[1] / n_vek[2] # Steigung in Z, sodass senkrecht zu Norm-Vek
    s_v = np.array([0.0, 1.0, a[0]])

    # Schnittgerade linker Link zur Ebene berechnen
    tan_30 = math.tan(math.degrees(30))
    a = (-n_vek[0] - n_vek[1] * tan_30) / n_vek[2]
    s_l = np.array([1, tan_30, a[0]])

    # Schnittgerade rechter Link zur Ebene berechnen
    a = (-n_vek[0] + n_vek[1] * tan_30) / n_vek[2]
    s_r = np.array([-1, tan_30, a[0]])


    # Endpunkte finden

    solver.s_v[:] = s_v # Steigungen im Solver aktualisieren
    solver.s_l[:] = s_l
    solver.s_r[:] = s_r

    stutzpunkt = np.array([5, 5, 5])

    result = solver.evaluate(stutzpunkt)    

    c = result[0] * s_v
    
    c = c

    
    

    solver = NumbaAbstande(s_r, s_v, s_l, 216.5064)
    solver.warmup()  # compile numba

    # initial guess for [tv, tl, tr]
    a0 = np.array([1.0, 1.0, 1.0])

    res = root(func_abstande, a0, args=(solver,), method="hybr")

    print("Success:", res.success)
    print("Solution:", res.x)   # → [tv, tl, tr]



def func_abstande(a, solver: NumbaAbstande):
    return solver.evaluate(a)

