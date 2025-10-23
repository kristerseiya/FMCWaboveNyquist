
import numpy as np

# system parameters
param_name = "sin_2e8_600_5"

# algorithms
# estimator_names = ['wn_snradjust_lattice', 'shortestpath_lattice']
# estimator_names = ['wn_snradjust_lattice', 'matchedfilter', 'freqavg'] 
# estimator_names = ['wn_snradjust_lattice_11_6',
#                    'wn_snradjust_lattice_10_6',
#                    'wn_snradjust_lattice_9_6',
#                    'wn_snradjust_lattice_8_6',
#                    'wn_snradjust_lattice_7_6',
#                    'wn_snradjust_lattice_6_6',
#                    'wn_snradjust_lattice_5_6',
#                    'wn_snradjust_lattice_4_6',
#                    'wn_snradjust_lattice_3_6',
#                    'wn_snradjust_lattice_11_4',
#                    'wn_snradjust_lattice_10_4',
#                    'wn_snradjust_lattice_9_4',
#                    'wn_snradjust_lattice_8_4',
#                    'wn_snradjust_lattice_7_4',
#                    'wn_snradjust_lattice_6_4',
#                    'wn_snradjust_lattice_5_4',
#                    'wn_snradjust_lattice_4_4',
#                    'wn_snradjust_lattice_3_4',
#                    'wn_snradjust_lattice_11_2',
#                    'wn_snradjust_lattice_10_2',
#                    'wn_snradjust_lattice_9_2',
#                    'wn_snradjust_lattice_8_2',
#                    'wn_snradjust_lattice_7_2',
#                    'wn_snradjust_lattice_6_2',
#                    'wn_snradjust_lattice_5_2',
#                    'wn_snradjust_lattice_4_2',
#                    'wn_snradjust_lattice_3_2',]
# estimator_names = ['wn_snradjust_lattice_faster_11_6',
#                    'wn_snradjust_lattice_faster_10_6',
#                    'wn_snradjust_lattice_faster_9_6',
#                    'wn_snradjust_lattice_faster_8_6',
#                    'wn_snradjust_lattice_faster_7_6',
#                    'wn_snradjust_lattice_faster_6_6',
#                    'wn_snradjust_lattice_faster_5_6',
#                    'wn_snradjust_lattice_faster_4_6',
#                    'wn_snradjust_lattice_faster_3_6',
#                    'wn_snradjust_lattice_faster_11_4',
#                    'wn_snradjust_lattice_faster_10_4',
#                    'wn_snradjust_lattice_faster_9_4',
#                    'wn_snradjust_lattice_faster_8_4',
#                    'wn_snradjust_lattice_faster_7_4',
#                    'wn_snradjust_lattice_faster_6_4',
#                    'wn_snradjust_lattice_faster_5_4',
#                    'wn_snradjust_lattice_faster_4_4',
#                    'wn_snradjust_lattice_faster_3_4',
#                    'wn_snradjust_lattice_faster_11_2',
#                    'wn_snradjust_lattice_faster_10_2',
#                    'wn_snradjust_lattice_faster_9_2',
#                    'wn_snradjust_lattice_faster_8_2',
#                    'wn_snradjust_lattice_faster_7_2',
#                    'wn_snradjust_lattice_faster_6_2',
#                    'wn_snradjust_lattice_faster_5_2',
#                    'wn_snradjust_lattice_faster_4_2',
#                    'wn_snradjust_lattice_faster_3_2',]
# estimator_names = ['wn_snradjust_lattice_faster', 
#                    'matchedfilter','matchedfilter_optim']
estimator_names = ['freqavg','matchedfilter_optim','matchedfilter2d_optim', 'wn_snradjust_lattice_faster'] 

# testing distance
# distance_arr = np.arange(5, 600, 0.1)
# velocity_arr = np.concatenate([np.arange(-75, 0, 1), np.arange(0, 75, 1)])
distance_arr = np.arange(1, 77, 0.1)
velocity_arr = np.arange(-75, 75, 6)

n_cycle = 1
