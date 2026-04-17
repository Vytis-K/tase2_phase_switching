import numpy as np
import matplotlib.pyplot as plt

data = np.load("synth_tas2_pulsing_v1.npz")
k = data["k"]
e = data["e"]

Irat0  = data["Irat0"]
IratA  = data["IratA"]
IratAB = data["IratAB"]
IratAB3= data["IratAB3"]
boundary = data["boundary_map"]

def show(img, title):
    plt.figure(figsize=(4.5,3.5))
    plt.imshow(img.T, origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel("x"); plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()

show(Irat0,  "Irat: pristine")
show(IratA,  "Irat: after pulse A")
show(IratAB, "Irat: after pulse B (erase A)")
show(IratAB3,"Irat: after +3 pulses B")

show(IratA-Irat0,   "ΔIrat (A - pristine)")
show(IratAB-IratA,  "ΔIrat (B - A)")
show(IratAB3-IratAB,"ΔIrat (+3B - B)")
show(boundary,      "domain-boundary proxy")

plt.show()
