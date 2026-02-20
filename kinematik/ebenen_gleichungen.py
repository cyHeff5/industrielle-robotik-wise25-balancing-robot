
import numpy as np
import math

def func_z_in_ebene(position, n_vek, d):
    
    z_punkt = (-position[0] * n_vek[0] - position[1] * n_vek[1] - d) / n_vek[2]

    return z_punkt

def func_d_fur_ebene(n_vek, punkt):
    # 0 = n_vektor[0] * stutze_v_u_pos[0] + n_vektor[1] * stutze_v_u_pos[1] + n_vektor[2] * platte_hohe_std + d


    d = -1 * np.sum(n_vek * punkt)


    return d