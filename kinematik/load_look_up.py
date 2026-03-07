from scipy.io import loadmat

look_up = loadmat("kinematik/look_up_erreichbarkeit.mat") # Gesamter mat file wir geladen

helper = look_up["h_min"]
h_min = helper[0, 0] # untere Grenze bleibt Konstant

h_max = look_up["h_max"]

helper = look_up["x_ref"]

x_min = helper[0, 0]
x_max = helper[0, -1]

look_up_data = {
    "h_min" : h_min,
    "h_max" : h_max,
    "x_min" : x_min,
    "x_max" : x_max
 }



