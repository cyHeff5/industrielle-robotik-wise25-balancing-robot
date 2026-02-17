import math

import numpy as np
try:
    from gpiozero import AngularServo
except ImportError:
    AngularServo = None

from kinematik.ebenen_gleichungen import *
from kinematik.solver import KinematikSolver


def func_kinematik_main(
    pos_arm_v,
    pos_arm_l,
    pos_arm_r,
    pos_ball,
    winkel_x,
    winkel_y,
    schwinge_o_l,
    schwinge_u_l,
    solver: KinematikSolver,
):
    ## Hauptablauf der Kinematik-Berechnung:

    winkel_x = math.radians(winkel_x)
    winkel_y = math.radians(winkel_y)

    #Rotationsmatrix berechnen:
    x_sin = math.sin(winkel_x)
    x_cos = math.cos(winkel_x)
    rx = np.array([[1, 0, 0], [0, x_cos, -x_sin], [0, x_sin, x_cos]])

    y_sin = math.sin(winkel_y)
    y_cos = math.cos(winkel_y)
    ry = np.array([[y_cos, 0, y_sin], [0, 1, 0], [-y_sin, 0, y_cos]])

    rot_mat = rx @ ry

    # Noramelenvektor drehen:
    n_vek = rot_mat @ np.array([[0], [0], [1]])

    # Schnittgerade vorderer Link zur Ebene berechen
    a = -n_vek[1] / n_vek[2]
    s_v = np.array([0.0, 1.0, a[0]])

    # Schnittgerade linker Link zur Ebene berechnen
    o = math.radians(30)
    tan_30 = math.tan(o)
    a = (-n_vek[0] - n_vek[1] * tan_30) / n_vek[2]
    s_l = np.array([-1.0, -tan_30, a[0]])

    # Schnittgerade rechter Link zur Ebene berechnen
    a = (-n_vek[0] + n_vek[1] * tan_30) / n_vek[2]
    s_r = np.array([1.0, -tan_30, a[0]])

    # Endpunkte finden
    solver.update_directions(s_v=s_v, s_l=s_l, s_r=s_r)
    x, _, solver_success, solver_iter = solver.solve()

    ep_v = x[0] * s_v
    ep_l = x[1] * s_l
    ep_r = x[2] * s_r

    # Hier Z Offset nach bedarf
    #--> Wir bleiben gerade immer um Standardhöhe
    ep_v[2] += 165
    ep_l[2] += 165
    ep_r[2] += 165

    # Feststellen: sind die Punkte erreichbar --> hier vereinfacht nur die Länge.
    # Geplant Liste mit Erreichbarkeit
    max_l = schwinge_o_l + schwinge_u_l
    ok = 1

    delta_l = func_laenge(ep_v, pos_arm_v)
    if delta_l > max_l:
        ok = 0
    delta_l = func_laenge(ep_l, pos_arm_l)
    if delta_l > max_l:
        ok = 0
    delta_l = func_laenge(ep_r, pos_arm_r)
    if delta_l > max_l:
        ok = 0

    # berechnen der Servowinkel:
    phi_servo_v = func_servo_winkel(pos_arm_v, ep_v, schwinge_o_l, schwinge_u_l)
    phi_servo_l = func_servo_winkel(pos_arm_l, ep_l, schwinge_o_l, schwinge_u_l)
    phi_servo_r = func_servo_winkel(pos_arm_r, ep_r, schwinge_o_l, schwinge_u_l)

    # neue Ebenengleichung berechnen:
    d = func_d_fur_ebene(n_vek, ep_v)

    return {
        "n_vek": n_vek,
        "d": d,
        "phi_servo_v": phi_servo_v,
        "phi_servo_l": phi_servo_l,
        "phi_servo_r": phi_servo_r,
        "ok": ok,
        "solver_success": solver_success,
        "solver_iter": solver_iter,
    }


def func_laenge(v1, v2):
    abstand = (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2
    abstand = abstand**0.5
    return abstand


def func_servo_winkel(servo_welle, endpunkt, schwinge_o_l, schwinge_u_l):
    #Funktion Berechnet von Endpunkt der Plattform und Servo Position den benötigten Einstellwinkel --> in Grad!

    winkel_welle = abs(math.atan(servo_welle[1] / servo_welle[0]))
    winkel_endpunkt = abs(math.atan((endpunkt[1] / endpunkt[0])))

    # 3D Koordinaten in einer Ebene auf diese in 2d reduzieren
    welle_2d = np.array([(servo_welle[0] ** 2 + servo_welle[1] ** 2) ** 0.5, servo_welle[2]])
    endpunkt_2d = np.array([(endpunkt[0] ** 2 + endpunkt[1] ** 2) ** 0.5, endpunkt[2]])
    l_ew_vek = endpunkt_2d - welle_2d
    l_ew = (l_ew_vek[0] ** 2 + l_ew_vek[1] ** 2) ** 0.5

    helper = (l_ew**2 + schwinge_u_l**2 - schwinge_o_l**2) / (2 * l_ew * schwinge_u_l)
    if abs(helper) <= 1:
        phi_strich = math.acos(helper)
    else:
        phi_strich = 0

    phi_lot = math.atan(l_ew_vek[0] / l_ew_vek[1])

    winkel = math.degrees(phi_lot + phi_strich)

    return winkel


def func_servo_drehen(servo, winkel, offset, modifier):
    #Stellt servo auf Winkel ein, Winkel wird mit Offset + modifier angepasst
    winkel = winkel * modifier + offset
    servo.angle = winkel
    return
