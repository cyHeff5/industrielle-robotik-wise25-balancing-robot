from kinematik.load_look_up import look_up_data
import numpy as np
import json
from pathlib import Path

_config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())


def func_erreichbarkeit_test(ep_v, ep_l, ep_r):
    v_data = func_min_max(ep_v)
    l_data = func_min_max(ep_l)
    r_data = func_min_max(ep_r)

    if (v_data["delta_z"] == 0) and (l_data["delta_z"] == 0) and (r_data["delta_z"] == 0):
        delta_z = 0
        alles_ok = 1

    elif (v_data["delta_z"] >= 0) and (l_data["delta_z"] >= 0) and (r_data["delta_z"] >= 0):
        delta_z = max(v_data["delta_z"], l_data["delta_z"], r_data["delta_z"])
        if (delta_z <= v_data["freiraum"][0]) and (delta_z <= l_data["freiraum"][0]) and (delta_z <= r_data["freiraum"][0]):
            alles_ok = 1
        else:
            alles_ok = 0  # nicht genug Spielraum -> Winkel abflachen

    elif (v_data["delta_z"] <= 0) and (l_data["delta_z"] <= 0) and (r_data["delta_z"] <= 0):
        delta_z = min(v_data["delta_z"], l_data["delta_z"], r_data["delta_z"])
        delta_z_inv = -delta_z
        if (delta_z_inv <= v_data["freiraum"][1]) and (delta_z_inv <= l_data["freiraum"][1]) and (delta_z_inv <= r_data["freiraum"][1]):
            alles_ok = 1
        else:
            alles_ok = 0  # nicht genug Spielraum -> Winkel abflachen
    else:
        alles_ok = 0  # ein Wert über, einer unter dem Limit -> Winkel abflachen

    if alles_ok == 0:
        delta_z = 0

    return {"delta_z": delta_z, "alles_ok": alles_ok}


def func_min_max(punkt):
    x_v = round((punkt[0]**2 + punkt[1]**2)**0.5)

    x_servo = 125  # X-Position der Servo-Welle [mm]
    min_diff = _config["min_diff"]

    z_min = ((min_diff)**2 - (x_v - x_servo)**2) ** 0.5

    if z_min < look_up_data["h_min"]:
        z_min = look_up_data["h_min"]

    ind = x_v - look_up_data["x_min"]
    z_max = look_up_data["h_max"][ind, 0]

    z_raum = np.array([z_max - punkt[2], punkt[2] - z_min])  # positiv = Platz vorhanden

    if z_raum[0] < 0:
        delta_muss = z_raum[0]
    elif z_raum[1] < 0:
        delta_muss = -z_raum[1]
    else:
        delta_muss = 0

    return {"delta_z": delta_muss, "freiraum": z_raum}
