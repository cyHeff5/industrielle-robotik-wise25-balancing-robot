from kinematik.load_look_up import look_up_data
import numpy as np


def func_erreichbarkeit_test(ep_v, ep_l, ep_r):


    # Wie viel Platz ist bei den Punkten nach oben oder nach unten? Wie muss die versetzt werden?
    v_data = func_min_max(ep_v)
    l_data = func_min_max(ep_l)
    r_data = func_min_max(ep_r)

    # Fall kein Punkt muss verschoben werden: 

    if (v_data["delta_z"] == 0) and (l_data["delta_z"] == 0) and (r_data["delta_z"] == 0):
        delta_z = 0
        alles_ok = 1
    
    elif (v_data["delta_z"] >= 0) and (l_data["delta_z"] >= 0) and (r_data["delta_z"] >= 0): 
        # Mindestens ein Wert muss hochgeschoben werden zur mindestehöhe
        delta_z = max(v_data["delta_z"], l_data["delta_z"], r_data["delta_z"])
        if (delta_z <= v_data["freiraum"][0]) and (delta_z <= l_data["freiraum"][0]) and (delta_z <= r_data["freiraum"][0]):
            alles_ok = 1 # der benötigte Versatz ist innerhalb des Spielraums der anderen Arme
        else: 
            alles_ok = 0 # die anderen Arme haben nicht genug Spielraum --> Winkel muss abgeflacht werden
    
    elif (v_data["delta_z"] <= 0) and (l_data["delta_z"] <= 0) and (r_data["delta_z"] <= 0): 
        # Mindestens ein Wert muss nach unten verschoben werden
        delta_z = min(v_data["delta_z"], l_data["delta_z"], r_data["delta_z"])
        delta_z_inv = -delta_z
        if (delta_z_inv <= v_data["freiraum"][1]) and (delta_z_inv <= l_data["freiraum"][1]) and (delta_z_inv <= r_data["freiraum"][1]):
            alles_ok = 1 # der benötigte Versatz ist innerhalb des Spielraums der anderen Arme
        else: 
            alles_ok = 0 # die anderen Arme haben nicht genug Spielraum --> Winkel muss abgeflacht werden
    else:
        alles_ok = 0 # Bedeutet: Ein Wert Oberhalb des Limits, einer Unterhalb, also muss Winkel flacher werden

    if alles_ok == 0:
        delta_z = 0

    return {"delta_z" : delta_z, "alles_ok": alles_ok}
    






def func_min_max(punkt):
     
    x_v =  round((punkt[0]**2 + punkt[1]**2)**0.5) # in 2d referenzebene überführen, gerundet

    ### WARNING HARDCODED INFO: 
    x_servo = 125 # X-Pos Servo Welle an 2D punkt
    min_diff = 120 # Länge zwischen Servo Well und Endpunkt

    # z-Min berechnen: 
    # brauchen mindestens länge min_diff zwischen Welle und Endpunkt
    z_min = ((min_diff)**2 - (x_v-x_servo)**2 ) ** 0.5

    if z_min < look_up_data["h_min"]: #Wenn kleiner als gesamt Minimum
        z_min = look_up_data["h_min"]

    ind = x_v - look_up_data["x_min"] # hier Grenzfall noch absichern
    z_max = look_up_data["h_max"][ind, 0]
    z_max =  z_max # Toleranzband
   
    z_raum = np.array([z_max - punkt[2], punkt[2] - z_min]) # Wenn Werte Positiv, dann ist Platz

    if z_raum[0] < 0: 
        delta_muss =  z_raum[0] # wenn addiert wird ergebnis negativ
    elif z_raum[1] < 0:
        delta_muss = - z_raum[1]
    else:
        delta_muss = 0
    return {"delta_z" : delta_muss, "freiraum": z_raum}

    



    
    


