import math
import time

import numpy as np
from gpiozero import AngularServo  # for servo control

from kinematik.ebenen_gleichungen import *
from kinematik.kinematik_main import func_kinematik_main
from kinematik.kinematik_main import func_servo_drehen
from kinematik.numba_abstande import NumbaAbstande
from kinematik.solver import KinematikSolver

t_ges = 0

# Required variables
platte_radius = 125  # [mm]
stutze_v_u_pos = np.array([0, platte_radius, 0])
delta_winkel = 120  # [deg]

platte_hohe_std = 165

schwinge_o_l = 179.2
schwinge_u_l = 70

# Servos
#servo_v = AngularServo(27, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)
modifier_v = 1
offset_v = 0
#func_servo_drehen(servo_v, 0, offset_v, modifier_v)

#servo_l = AngularServo(17, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)
modifier_l = 1
offset_l = 0
#func_servo_drehen(servo_l, 0, offset_l, modifier_l)

#servo_r = AngularServo(22, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)
modifier_r = 1
offset_r = 0
#func_servo_drehen(servo_r, 0, offset_r, modifier_r)

# Pre-calculations
delta_winkel = math.radians(delta_winkel - 90)
stutze_l_u_pos = np.array([-math.cos(delta_winkel) * platte_radius, -math.sin(delta_winkel) * platte_radius, 0])
stutze_r_u_pos = np.array([stutze_l_u_pos[0] * -1, stutze_l_u_pos[1], 0])

# First loop setup
n_vektor = np.array([0, 0, 1])
d = func_d_fur_ebene(n_vektor, [stutze_v_u_pos[0], stutze_v_u_pos[1], platte_hohe_std])

# Solver setup
s_r = np.array([1.0, -0.4, 0.9], dtype=np.float64)
s_v = np.array([0.8, 0.6, -0.7], dtype=np.float64)
s_l = np.array([-0.5, 0.3, 1.1], dtype=np.float64)

abstande = NumbaAbstande(s_r=s_r, s_l=s_l, s_v=s_v, l=216.5064)
abstande.warmup()

# Choose solver method: "newton" or "scipy"
solver_method = "scipy"
solver_positionen = KinematikSolver(abstande=abstande, method=solver_method, tol=1e-3, max_iter=50)

# Main loop simulation
ball_pos = np.array([15, 20, -100])
ball_pos[2] = func_z_in_ebene(ball_pos, n_vektor, d)

n = 0 #counter
pos_ref = np.array([0, 0, 165]) # std Referenzpunkt für Höhe
while n < 8:
        
    match n:
        case 0:
            winkel_x    = 0
            winkel_y    = 0
            pos_ref[2]  = 140
        case 1: 
            winkel_x    = -7
            winkel_y    = 0
            pos_ref[2]  = 165 
        case 2: 
            winkel_x    = -7
            winkel_y    = 0
            pos_ref[2]  = 165
        case 3: 
            winkel_x    = 0
            winkel_y    = 0
            pos_ref[2]  = 165
        case 4: 
            winkel_x    = 0
            winkel_y    = -7
            pos_ref[2]  = 165
        case 5: 
            winkel_x    = 0
            winkel_y    = 7
            pos_ref[2]  = 165
        case 6: 
            winkel_x    = 7
            winkel_y    = -7
            pos_ref[2]  = 165
        case 6: 
            winkel_x    = 0
            winkel_y    = 0
            pos_ref[2]  = 165


   
    kinematik = func_kinematik_main(
        stutze_v_u_pos,
        stutze_l_u_pos,
        stutze_r_u_pos,
        ball_pos,
        winkel_x,
        winkel_y,
        schwinge_o_l,
        schwinge_u_l,
        solver_positionen,
    )
    #func_servo_drehen(servo_v, kinematik["phi_servo_v"], offset_v, modifier_v) # Servos ansteuern
    #func_servo_drehen(servo_l, kinematik["phi_servo_l"], offset_l, modifier_l)
    #func_servo_drehen(servo_r, kinematik["phi_servo_r"], offset_r, modifier_r)
    #time.sleep(5)
    n += 1



    d = kinematik["d"] # Information aus Kinematik für nächsten Loop
    n_vektor = kinematik["n_vek"]




