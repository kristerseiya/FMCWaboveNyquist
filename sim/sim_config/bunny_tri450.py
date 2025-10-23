
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# system parameters
param_name = "tri_2e7_450_4"

# algorithms
# estimator_names = ['wn_snradjust_lattice', 'shortestpath_lattice']
# estimator_names = ['wn_snradjust_lattice', 'matchedfilter', 'freqavg'] 
# estimator_names = ['wn_snradjust_lattice', 'lorentz', 'maxpd'] 
estimator_names = ['maxpd',] 

# testing distance

depth = Image.open('bunnydepth4.png')
depth = np.array(depth)
distance_arr = depth[depth>0]
distance_arr -= 290
n_cycle = 1

