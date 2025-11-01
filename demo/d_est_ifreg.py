
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import numpy as np
import matplotlib.pyplot as plt

import fmcw_sys
import time_freq
import estimators
from estimators import presets
import time

param_name = 'tri_2e8_600_5'
n_cycle = 1

meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join(this_dir, '..', 'fmcw_sys', 'params.json'), param_name)
meas_prop.sample_rate = 1e10
dist_true = 120
vel_true = 0
tau_true = 2*dist_true/3e8
stft_window_size = 64

np.random.seed(7151)
meas_prop.assume_zero_velocity = True
sample_rate = meas_prop.get_sample_rate()
T = meas_prop.get_chirp_length()
t = np.arange(0, 2*T*n_cycle, 1./sample_rate)
fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
signal, second_output = fmcw_meas.generate(dist_true, vel_true, t, include_shot_noise=True)
print(len(signal))

now = time.time()
key = {'gridgen_type':'optimal', 'method':'LBFGS', 'init_step': 'none', 'ignore_quadrature':False, 'snr_adjustment':True}
estimator = estimators.IFRegressor(meas_prop, **key)
for _ in range(1):
    x_hat = estimator.estimate(t, signal, second_output)
print(time.time()-now)
print(estimator.total_n_evals)

print('    True: {:.4f}m, {:.4f}m/s'.format(dist_true, vel_true))
print('Estimate: {:.4f}m, {:.4f}m/s'.format(x_hat[0], x_hat[1]))

now = time.time()
for _ in range(1):
    estimator = estimators.LorentzianRegressor(meas_prop, method="LBFGS")
    x_hat = estimator.estimate(t, signal, second_output)
print(time.time()-now)

print('    True: {:.4f}m, {:.4f}m/s'.format(dist_true, vel_true))
print('Estimate: {:.4f}m, {:.4f}m/s'.format(x_hat[0], x_hat[1]))

plt.show()