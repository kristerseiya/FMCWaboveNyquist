
import numpy as np

# system parameters
param_name = "tri_2e9_600_5"

# algorithms
# estimator_names = ['wn_snradjust_lattice', 'shortestpath_lattice']
# estimator_names = ['wn_snradjust_lattice', 'lorentz', 'maxpd']
estimator_names = ['matchedfilter','matchedfilter_optim']

# testing distance
# distance_arr = np.arange(5, 600, 0.1)
# velocity_arr = np.concatenate([np.arange(-75, 0, 0.1), np.arange(0, 75, 0.1)])
distance_arr = np.arange(5, 600, 0.1)
velocity_arr = np.concatenate([np.arange(-75, 0, 1), np.arange(0, 75, 1)])
# distance_arr = np.arange(5, 600, 200)
# velocity_arr = np.arange(-75, 75 , 100)
n_cycle = 1
