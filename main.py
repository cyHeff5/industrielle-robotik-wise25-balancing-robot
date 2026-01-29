
import numpy as np
import math
from gpiozero import AngularServo #für Servo Ansteuerung

from ebenen_gleichungen import *
from kinematik import *



# benötigte Variablen: 

platte_radius      = 125 # in [mm]
stutze_v_u_pos     = np.array([0, platte_radius, 0]) # Position Stütze vorne unten --> anderen beiden errrechnen sich
delta_winkel       = 120  # Winkel in dem die Stüetzen stehen [grad]

platte_hohe_std    = 165 #  Ausgangshöhe der Platte über Servowelle bei 0°

schwinge_o_l = 179.2
schwinge_u_l = 70

# Servos definieren

#Servo vorne:
servo_v     = AngularServo(18, min_angle= 0, max_angle=180, min_pulse_width= 0.0005, max_pulse_width=0.0025)
modifier_v  = 1 #Faktor für Winkel
offset_v    = 0 # Offset für Winkel
#Servo links:
servo_l     = AngularServo(18, min_angle= 0, max_angle=180, min_pulse_width= 0.0005, max_pulse_width=0.0025)
modifier_l  = 1 #Faktor für Winkel
offset_l    = 0 # Offset für Winkel
#Servo rechts:
servo_r     = AngularServo(18, min_angle= 0, max_angle=180, min_pulse_width= 0.0005, max_pulse_width=0.0025)
modifier_r  = 1 #Faktor für Winkel
offset_r    = 0 # Offset für Winkel





#vorab Berechnungen
delta_winkel   = math.radians(delta_winkel - 90) # [rad] wie weit die unteren Stüetzen von der x achse aus verdreht sind
stutze_l_u_pos = np.array([-math.cos(delta_winkel) * platte_radius, -math.sin(delta_winkel) * platte_radius, 0 ])
stutze_r_u_pos = np.array([stutze_l_u_pos[0] * -1, stutze_l_u_pos[1], 0])

# ersten Loop vorbereiten --> wir gehen von ebenen System aus
n_vektor = np.array([0, 0, 1]) #Normalenvektor der Ebene'
# Ebene wird beschrieben durch: n_x * x + n_y * y + n_x * x = d mit n_? Anteil des Normalenvektor
d = func_d_fur_ebene(n_vektor, [stutze_v_u_pos[0], stutze_v_u_pos[1], platte_hohe_std]) # d für Ebene berechnen 


############# Main/Berechnungen: ############# 

 ###Loop start

## Bildverarbeitung

ball_pos = np.array([15, 20, -100]) # Ballposition aus Reglung --> kann nur x-y Position bestimmen

## Ballkoordinate vervollständigen
ball_pos[2] = func_z_in_ebene(ball_pos, n_vektor, d)

## Regler berechnungen

winkel_x = 0 # Angabe in [Grad]
winkel_y = 0


## Kinematik berechnen --> ergibt Servowinkel
kinematik = func_kinematik_main(stutze_v_u_pos, stutze_l_u_pos, stutze_r_u_pos, ball_pos, 15, -10, schwinge_o_l, schwinge_u_l)

d         = kinematik["d"] # Informationen für die nächste Loop-Iteration
n_vektor  = kinematik["n_vek"] 

## Servos ansteuern ##

func_servo_drehen(servo_v, kinematik['phi_servo_v'], offset_v, modifier_v)
func_servo_drehen(servo_l, kinematik['phi_servo_l'], offset_l, modifier_l)
func_servo_drehen(servo_r, kinematik['phi_servo_r'], offset_r, modifier_r)
 ### Loop end