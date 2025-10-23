import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(os.path.join(this_dir, '..', '..'))

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import fmcw_sys
import utils

# load parameters
config_filepath = utils.PARAMS_PATH
config_name = 'tri_2e8_600_5'
meas_prop = fmcw_sys.import_meas_prop_from_config(config_filepath, config_name)
meas_prop.reflectance = 1e-4
modf = fmcw_sys.modulation.get_modulation(meas_prop)

# system parameters
linewidth = meas_prop.get_linewidth()
sample_rate = meas_prop.get_sample_rate()
reflectance = meas_prop.reflectance
detector_effectivity = meas_prop.detector_effectivity # A/W
transmitted_power = meas_prop.transmitted_power # W
detector_effective_area = meas_prop.detector_effective_area # m^2
lo_power = transmitted_power
q = 1.6e-19

# simulation parameters
# distance_arr = np.arange(1, meas_prop.get_max_d(), 0.05)
distance_arr = np.arange(1, meas_prop.get_max_d(), 1)
n_sample = 10000
polyfit_deg = 7 # must be even
frechet_var = np.zeros((len(distance_arr)))
frechet_corr = np.zeros((len(distance_arr)))

pbar = tqdm(total=len(distance_arr), leave=True)

for i, distance in enumerate(distance_arr):

    rx_power =  transmitted_power * detector_effective_area / (np.pi*distance**2) * reflectance
    # snr = (8 * detector_effectivity**2 *lo_power *rx_power)/(2*q*detector_effectivity*(lo_power+rx_power))
    # mean = 2 * detector_effectivity * np.sqrt(lo_power * rx_power)
    # noise = (np.random.randn(n_sample) + 1j * np.random.randn(n_sample)) * np.sqrt(q*detector_effectivity*(lo_power+rx_power))
    # x = mean + noise
    snr = (2 * detector_effectivity**2 *lo_power *rx_power)/(q*detector_effectivity*(lo_power+rx_power))
    mean = np.sqrt(2) * detector_effectivity * np.sqrt(lo_power * rx_power)
    noise = (np.random.randn(n_sample) + 1j * np.random.randn(n_sample)) * np.sqrt(0.5*q*detector_effectivity*(lo_power+rx_power))
    x = mean + noise

    diff_projected_noise = np.angle(x[1:]*np.conj(x[:-1]))
    diff_projected_noise = np.mod(diff_projected_noise + np.pi, 2*np.pi) - np.pi
    # diff_projected_noise = np.angle(x)
    # diff_projected_noise = np.mod(diff_projected_noise + np.pi, 2*np.pi) - np.pi
    frechet_var[i] = np.mean(diff_projected_noise**2)
    frechet_corr[i] = np.mean(diff_projected_noise[1:]*diff_projected_noise[:-1])
    pbar.update(1)

pbar.close()

A = np.stack([distance_arr**k for k in range(0, polyfit_deg+1, 2)], axis=-1)
# D = np.diag(1/(distance_arr+0.1))
D = np.diag(np.exp(-distance_arr/80))
frechet_var_polyfit_coefs_evens = np.linalg.inv(A.T @ D @ A) @ A.T @ D @ frechet_var
frechet_var_polyfit_coefs = np.zeros((len(frechet_var_polyfit_coefs_evens)*2-1))
for i in range(0, len(frechet_var_polyfit_coefs_evens)):
    frechet_var_polyfit_coefs[2*i] = frechet_var_polyfit_coefs_evens[i] 
frechet_var_polyfit_coefs = frechet_var_polyfit_coefs[::-1]
# frechet_var_polyfit_coefs = np.polyfit(distance_arr, frechet_var, deg=6)
frechet_var_polyfit = np.poly1d(frechet_var_polyfit_coefs)

A = np.stack([distance_arr**k for k in range(0, polyfit_deg+1+8, 2)], axis=-1)
D = np.diag(np.exp(-distance_arr/600))
frechet_corr_polyfit_coefs_evens = np.linalg.inv(A.T @ D @ A) @ A.T @ D @ frechet_corr
frechet_corr_polyfit_coefs = np.zeros((len(frechet_corr_polyfit_coefs_evens)*2-1))
for i in range(0, len(frechet_corr_polyfit_coefs_evens)):
    frechet_corr_polyfit_coefs[2*i] = frechet_corr_polyfit_coefs_evens[i] 
frechet_corr_polyfit_coefs = frechet_corr_polyfit_coefs[::-1]
# frechet_var_polyfit_coefs = np.polyfit(distance_arr, frechet_var, deg=6)
frechet_corr_polyfit = np.poly1d(frechet_corr_polyfit_coefs)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(131)
ax1.plot(distance_arr, frechet_var)
ax1.plot(distance_arr, frechet_var_polyfit(distance_arr))
ax2 = fig1.add_subplot(132)
ax2.plot(distance_arr, frechet_corr)
ax2.plot(distance_arr, frechet_corr_polyfit(distance_arr))
ax3 = fig1.add_subplot(133)
ax3.plot(distance_arr, -0.5*np.ones_like(distance_arr))
ax3.plot(distance_arr, frechet_corr_polyfit(distance_arr)/frechet_var_polyfit(distance_arr))
plt.show()

np.savez(os.path.join(this_dir,'diff_projected_noise_frechet_var_highsnr'), distance=distance_arr, frechet_var=frechet_var, 
         frechet_var_polyfit_coefs=frechet_var_polyfit_coefs, 
         frechet_corr_polyfit_coefs=frechet_corr_polyfit_coefs, 
         n_sample=n_sample)