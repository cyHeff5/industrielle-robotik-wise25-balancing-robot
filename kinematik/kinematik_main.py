import numpy as np
import math

from gpiozero import AngularServo

from kinematik.ebenen_gleichungen import *
from kinematik.erreichbarkeit import func_erreichbarkeit_test


from kinematik.numba_abstande import NumbaAbstande # für den Solver
from scipy.optimize import root

def func_kinematik_main(pos_arm_v, pos_arm_l, pos_arm_r, ref_point, winkel_x, winkel_y, schwinge_o_l, schwinge_u_l, solver):

    ## Hauptablauf der Kinematik-Berechnung:

    winkel_x = math.radians(winkel_x)
    winkel_y = math.radians(winkel_y)

    alles_ok = 0

    while alles_ok == 0:
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
        o = math.radians(30)
        tan_30 = math.tan(o)
        a = (+n_vek[0] +n_vek[1] * tan_30) / n_vek[2]
        s_l = np.array([-1, -tan_30, a[0]])

        # Schnittgerade rechter Link zur Ebene berechnen
        a = (-n_vek[0] + n_vek[1] * tan_30) / n_vek[2]
        s_r = np.array([1, -tan_30, a[0]])


        # Endpunkte finden

        solver.update_directions(s_v=s_v, s_l=s_l, s_r=s_r)
        x, _, solver_success, solver_iter = solver.solve()

        ep_v = x[0] * s_v #Punkte mit Solver ergebnis berechnen
        ep_l = x[1] * s_l
        ep_r = x[2] * s_r

        # Hier Z Offset: ref_point bleibt auf konstanter Höhe

        d = func_d_fur_ebene(n_vek, ref_point)
        test = func_z_in_ebene(ref_point, n_vek, d)
        #--> Wir bleiben gerade immer um Standardhöhe
        ep_v[2] = func_z_in_ebene(ep_v, n_vek, d)
        ep_l[2] = func_z_in_ebene(ep_l, n_vek, d)
        ep_r[2] = func_z_in_ebene(ep_r, n_vek, d)

        #Erreichbarkeit der Endpunkte Testen:

        erreichbarkeit = func_erreichbarkeit_test(ep_v, ep_l, ep_r)
        alles_ok = erreichbarkeit["alles_ok"]

        if alles_ok == 1: # z-Verschiebung verwenden
            ep_v[2] += erreichbarkeit["delta_z"]
            ep_l[2] += erreichbarkeit["delta_z"]
            ep_r[2] += erreichbarkeit["delta_z"]
        else: # Sollte die Platfform nicht in die Erreichbarkeit passen, dann abflachen
            winkel_x = 0.90 * winkel_x
            winkel_y = 0.90 * winkel_y
    
    # Ende Korrekturloop
                

        
    # Berechnen der Servowinkel: 

    phi_servo_v = func_servo_winkel(pos_arm_v, ep_v, schwinge_o_l, schwinge_u_l)
    phi_servo_l = func_servo_winkel(pos_arm_l, ep_l, schwinge_o_l, schwinge_u_l)
    phi_servo_r = func_servo_winkel(pos_arm_r, ep_r, schwinge_o_l, schwinge_u_l)


    # neue Ebenengleichung berechnen:
    d = func_d_fur_ebene(n_vek, ep_v) #D für Ebenengleichung Ausrechnen


    



    return {'n_vek': n_vek, 'd' : d ,'phi_servo_v': phi_servo_v, 'phi_servo_l': phi_servo_l, 'phi_servo_r': phi_servo_r }


def func_laenge(v1, v2):

    abstand = (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2 
    abstand = abstand**0.5
    return abstand
    


def func_abstande(a, solver: NumbaAbstande):
    return solver.evaluate(a)

def func_servo_winkel(servo_welle, endpunkt, schwinge_o_l, schwinge_u_l):
    #Funktion Berechnet von Endpunkt der Plattform und Servo Position den benötigten Einstellwinkel --> in Grad!

    # Wir rechnen Testweise mal die Winkel aus: Sind die colinear?
    #winkel_welle    = abs(math.atan(servo_welle[1]/servo_welle[0]))
    #winkel_endpunkt = abs(math.atan((endpunkt[1]/endpunkt[0])))

    # welle_2d    = np.array([servo_welle[1] * math.sin(winkel_welle), servo_welle[2]]) # 3D Koordinaten in einer Ebene auf diese in 2d reduzieren
    #endpunkt_2d = np.array([endpunkt[1] * math.sin(winkel_welle), endpunkt[2]])
    welle_2d    = np.array([(servo_welle[0]**2 + servo_welle[1]**2)**0.5, servo_welle[2]])
    endpunkt_2d = np.array([(endpunkt[0]**2 + endpunkt[1]**2)**0.5, endpunkt[2]])
    l_ew_vek = endpunkt_2d - welle_2d # Vektor von Welle zu Endpunkt
    l_ew     = (l_ew_vek[0]**2 + l_ew_vek[1]**2)**0.5

    helper = (l_ew**2 + schwinge_u_l**2 - schwinge_o_l**2) / (2* l_ew * schwinge_u_l)
    if abs(helper) <= 1:
     phi_strich = math.acos(helper)
     phi_lot = math.atan(l_ew_vek[0] / l_ew_vek[1])
     winkel = math.degrees(phi_lot + phi_strich) # Winkel in [°]
    else: # Falls Endpunkt nicht erreichbar
     winkel = 130





    return winkel #[°]



def func_servo_drehen(servo, winkel, offset, modifier):
    #Stellt servo auf Winkel ein, Winkel wird mit Offset + modifier angepasst
    winkel = winkel * modifier + offset

    if winkel > 180:
       winkel = 180
    if winkel < 0:
        winkel = 0
    servo.angle = winkel
    return

