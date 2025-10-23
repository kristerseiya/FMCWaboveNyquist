
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
from scipy.fftpack import fft, ifft

# param_name = 'tri_2e8_1000_5'
# param_name = 'tri_2e7_300_2'
param_name = 'tri_2e8_600_5'
# param_name = 'stair_2e8_600_5'
n_cycle = 1

meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join(this_dir, '..', 'fmcw_sys', 'params.json'), param_name)
# meas_prop.assume_zero_velocity = True
# meas_prop.complex_available = True
# meas_prop.bandwidth = 1e8
# meas_prop.Tchirp = 1e-5
# meas_prop.reflectance = 1e-2
# meas_prop.modulation = 'hardsteep'
# meas_prop.modulation = 'smoothstairs'
meas_prop.linewidth = 1e1
# dist_true = 590.5
dist_true = 60
vel_true = -3

tau_true = 2*dist_true/3e8
stft_window_size = 64

# np.random.seed(10010)
# np.random.seed(27635804)
meas_prop.assume_zero_velocity = False
sample_rate = meas_prop.get_sample_rate()
T = meas_prop.get_chirp_length()
t = np.arange(0, 2*T*n_cycle, 1./sample_rate)
fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
signal, second_output = fmcw_meas.generate(dist_true, vel_true, t, include_shot_noise=True)

# fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
# sample_rate = meas_prop.get_sample_rate()
# bandwidth = meas_prop.get_bandwidth()
# if bandwidth > sample_rate:
#     K = int(np.ceil(bandwidth/sample_rate))
# else:
#     K = 1
# K = 2
# print(K)
# T = meas_prop.get_chirp_length()
# ta = np.arange(0, 2*T, 1/sample_rate/K)
# xa = np.zeros((len(ta)), dtype=signal.dtype)
# N = len(signal)
# xa[0:N*K:K] = signal
# phase_tx = fmcw_meas.generate_phase(ta, 0, 0)
# if meas_prop.is_complex_available():
#     filter_output = np.abs(ifft(fft(xa*np.exp(-1j*phase_tx))*np.conj(fft(np.exp(-1j*phase_tx))) ))
#     filter_output = filter_output[:N*K]
# else:
#     filter_output = np.abs(ifft(fft(xa*np.exp(-1j*phase_tx))*np.conj(fft(np.exp(-1j*phase_tx))) ) + ifft(fft(xa*np.exp(1j*phase_tx))*np.conj(fft(np.exp(1j*phase_tx))) ))
#     filter_output = filter_output[:N*K]

# max_idx = np.argmax(np.abs(filter_output))
# tau = ta[max_idx]
now = time.time()
for _ in range(1):
    estimator = estimators.MathedFilterDelayEstimator2D(meas_prop, optimize=False)
    x_hat = estimator.estimate(t, signal, second_output)
print(time.time()-now)
# d = np.linspace(538, 542, 500)
# v = np.zeros_like(d)
# a = estimator.if_likelihood.evaluate(np.stack([d,v], axis=1))
# b = estimator.if_likelihood.compute_gradient(np.stack([d,v], axis=1))
# plt.subplot(1,3,1)
# plt.plot(d, a)
# plt.subplot(1,3,2)
# plt.plot(d, b[:,0])
# plt.subplot(1,3,3)
# plt.plot(d[1:], a[1:]-a[:-1])

print('    True: {:.4f}m, {:.4f}m/s'.format(dist_true, vel_true))
print('Estimate: {:.4f}m, {:.4f}m/s'.format(x_hat[0], x_hat[1]))
plt.show()