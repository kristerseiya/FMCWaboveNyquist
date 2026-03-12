
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(os.path.join(this_dir, '..', '..'))

import numpy as np
import datetime
import argparse
from importlib import import_module

import utils
from estimators import presets, IFRegressor
import fmcw_sys

import time


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--extra_name', type=str, default='')
    parser.add_argument('--n_sim', type=int, default=1)
    args = parser.parse_args()

    # import paramters
    config = import_module('.'+args.config, package='sim_config')

    if args.seeds is None:
        distance_arr = config.distance_arr
        velocity_arr = config.velocity_arr
        n_cycle = config.n_cycle
        n_simulation = args.n_sim
        measseeds_arr = (np.random.rand(len(distance_arr), len(velocity_arr), n_simulation)*(2**32-1)).astype(int)
    else:
        sim_data = np.load(os.path.join(this_dir,'dv_est_sim_results',args.seeds+'_seeds.npz'))
        measseeds_arr = sim_data['seeds']
        distance_arr = sim_data['distance']
        velocity_arr = sim_data['velocity']
        n_simulation = int(sim_data['n_simulation'])
        n_cycle = sim_data['n_cycle']

    n_cycle = config.n_cycle
    meas_prop = fmcw_sys.import_meas_prop_from_config(utils.PARAMS_PATH, config.param_name)
    sample_rate = meas_prop.get_sample_rate()
    T = meas_prop.get_chirp_length()
    t = np.arange(0, 2*T*n_cycle, 1./sample_rate)
    
    meas_prop.assume_zero_velocity = False
    fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
    
    B = meas_prop.get_bandwidth()
    sample_rate = meas_prop.get_sample_rate()
    T = meas_prop.get_chirp_length()


    # get estimators
    n_estimator = len(config.estimator_names)
    estimators = presets.get_estimators(meas_prop, config.estimator_names)
    
    # create simulation loop
    x_hat_results = np.zeros((n_estimator, len(distance_arr), len(velocity_arr), n_simulation, 2))

    comp_times = np.zeros((n_estimator, len(distance_arr), len(velocity_arr), n_simulation))
    n_evals = np.zeros((n_estimator, len(distance_arr), len(velocity_arr), n_simulation))
     
    for d_idx in range(len(distance_arr)):
        
        dist_true = distance_arr[d_idx]
        
        for v_idx in range(len(velocity_arr)):
        
            velo_true = velocity_arr[v_idx]

            for n in range(n_simulation):

                np.random.seed(measseeds_arr[d_idx, v_idx, n])
                signal, second_output = fmcw_meas.generate(dist_true, velo_true, t)

                for i, estimator in enumerate(estimators):
                    start = time.time()
                    if isinstance(estimator, presets.Estimator):
                        x_hat = estimator.estimate(t, signal, second_output)
                    elif isinstance(estimator, presets.OracleEstimator):
                        x_hat = estimator.estimate(np.array([dist_true, 0]), t, signal, second_output)
                    x_hat_results[i,d_idx,v_idx,n,0] = x_hat[0]
                    x_hat_results[i,d_idx,v_idx,n,1] = x_hat[1]
                    done = time.time()
                    comp_times[i,d_idx,v_idx,n] = done-start
                    n_evals[i,d_idx,v_idx,n] = estimator.n_eval
                    
            print(f"{d_idx*len(velocity_arr)+v_idx+1:d}/{len(distance_arr)*len(velocity_arr):d}")


    if not os.path.exists(os.path.join(this_dir, 'dv_est_sim_results')):
        os.makedirs(os.path.join(this_dir, 'dv_est_sim_results'))
    results_dir = os.path.join(this_dir, 'dv_est_sim_results')

    # save results
    # generate file key
    now = datetime.datetime.now()
    datetime_str = now.strftime("%m%d%y%H%M%S")
    if args.seeds is not None:
        results_filename = f'{datetime_str:s}_estimates_{args.seeds:s}_seeds_{args.extra_name:s}.npz'
        results_filename = os.path.join(results_dir, results_filename)
    else:
        seeds_filename = datetime_str+'_seeds.npz'
        seeds_filename = os.path.join(results_dir, seeds_filename)
        np.savez(seeds_filename,  description=args.description, 
                 distance=distance_arr, seeds=measseeds_arr, n_simulation=n_simulation)
        results_filename = f'{datetime_str:s}_estimates_{datetime_str:s}_seeds_{args.extra_name:s}.npz'
        results_filename = os.path.join(results_dir, results_filename)

    # save estimator output
    estimator_name_keys = list()
    estimator_name_keys_comp_time = list()
    estimator_name_keys_n_evals = list()
    for estimator_name in config.estimator_names:
        estimator_name_keys.append('estimator_'+estimator_name)
        estimator_name_keys_comp_time.append('estimator_'+estimator_name+'_comp_time')
        estimator_name_keys_n_evals.append('estimator_'+estimator_name+'_n_evals')
        
    np.savez(results_filename, description=args.description, 
             param_name=config.param_name, **(meas_prop.__dict__), 
             distance=distance_arr, velocity=velocity_arr, n_simulation=n_simulation, n_cycle=n_cycle, **dict(zip(estimator_name_keys, x_hat_results)),
             **dict(zip(estimator_name_keys_comp_time, comp_times)),
             **dict(zip(estimator_name_keys_n_evals, n_evals)))