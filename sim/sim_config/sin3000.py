
import numpy as np

# system parameters
param_name = "sin_2e8_3000_5"

# algorithms
# estimator_names = ['wn_snradjust_lattice', 'shortestpath_lattice']
# estimator_names = ['wn_snradjust_lattice', 'matchedfilter', 'freqavg'] 
estimator_names = ['freqavg',] 


# testing distance
# distance_arr = np.arange(5, 600, 0.1)
# velocity_arr = np.concatenate([np.arange(-75, 0, 0.1), np.arange(0, 75, 0.1)])
distance_arr = np.arange(5, 600, 0.1)
velocity_arr = np.concatenate([np.arange(-75, 0, 1), np.arange(0, 75, 1)])

n_cycle = 1
