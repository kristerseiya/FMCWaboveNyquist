
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(os.path.join(this_dir, '..', '..'))

import numpy as np
import datetime
import argparse
from importlib import import_module
import multiprocessing
import itertools

import utils
import fmcw_sys
from estimators import presets


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--n_sim', type=int, default=1)
    args = parser.parse_args()

    # import paramters
    config = import_module('.'+args.config, package='sim_config')

    n_simulation = args.n_sim
    distance_arr = config.distance_arr
    # measseeds_arr_buffer = multiprocessing.RawArray(np.ctypeslib.as_ctypes_type(int), len(distance_arr)*n_simulation)
    # measseeds_arr = np.frombuffer(measseeds_arr_buffer, dtype=int).reshape((len(distance_arr), n_simulation))
    measseeds_arr = (np.random.rand(len(distance_arr), n_simulation)*(2**32-1)).astype(int)

    if not os.path.exists(os.path.join(this_dir, 'd_est_sim_results')):
        os.makedirs(os.path.join(this_dir, 'd_est_sim_results'))
    results_dir = os.path.join(this_dir, 'd_est_sim_results')

    # save results
    # generate file key
    now = datetime.datetime.now()
    datetime_str = now.strftime("%m%d%y%H%M%S")

    seeds_filename = datetime_str+'_seeds.npz'
    seeds_filename = os.path.join(results_dir, seeds_filename)
    np.savez(seeds_filename, 
                distance=distance_arr, seeds=measseeds_arr, n_simulation=n_simulation)
