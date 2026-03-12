
import numpy as np

# system parameters
param_name = "tri_2e8_600_5"

# algorithms
# estimator_names = ['wn_snradjust_lattice', 'shortestpath_lattice']
# estimator_names = ['wn_snradjust_lattice', 'lorentz', 'maxpd']
estimator_names = ['wn_snradjust_lattice_faster',
                   'lorentz_faster', 'maxpd',
                   'matchedfilter_optim',
                   'matchedfilter2d_optim']

# testing distance
# distance_arr = np.arange(5, 5+77, 7)
# velocity_arr = np.arange(-75, 75, 14)
distance_arr = np.arange(1, 120, 3.5)
velocity_arr = np.arange(-75, 75, 7)
n_cycle = 1
