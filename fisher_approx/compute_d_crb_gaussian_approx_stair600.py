
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(os.path.join(this_dir, '..', '..'))

import numpy as np
from tqdm import tqdm
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

import fmcw_sys
import utils

# load parameters
config_filepath = utils.PARAMS_PATH
config_name = 'stair_2e8_600_5'
meas_prop = fmcw_sys.import_meas_prop_from_config(config_filepath, config_name)
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
T = meas_prop.get_chirp_length()

# simulation parameters
t = np.arange(0, 2*T, 1/sample_rate)
t = 0.5 * (t[1:] + t[:-1])
distance_arr = np.arange(1, 600, 1)
precomputed_data = np.load(os.path.join(this_dir, 'diff_projected_noise_frechet_var_highsnr.npz'))
diff_prdjected_noise_freceht_var_coefs = precomputed_data['frechet_var_polyfit_coefs']
diff_prdjected_noise_freceht_var = np.poly1d(diff_prdjected_noise_freceht_var_coefs)(distance_arr)
d_diff_prdjected_noise_freceht_var = utils.compute_polynomial_derivative(distance_arr, diff_prdjected_noise_freceht_var_coefs)

diff_prdjected_noise_freceht_corr_coefs = precomputed_data['frechet_corr_polyfit_coefs']
diff_prdjected_noise_freceht_corr = np.poly1d(diff_prdjected_noise_freceht_corr_coefs)(distance_arr)
d_diff_prdjected_noise_freceht_corr = utils.compute_polynomial_derivative(distance_arr, diff_prdjected_noise_freceht_corr_coefs)

d_crb_approx_full = np.zeros((len(distance_arr)))
d_crb_approx = np.zeros((len(distance_arr)))
d_crb_approx_iid = np.zeros((len(distance_arr)))

pbar = tqdm(total=len(distance_arr), leave=True)
for i, distance in enumerate(distance_arr):
    cov_1st_row = np.zeros((len(t)))
    cov_1st_row[0] = 4*np.pi*linewidth/sample_rate
    cov_1st_row[0] += diff_prdjected_noise_freceht_var[i]
    # cov_1st_row[1] = -1/2*diff_prdjected_noise_freceht_var[i]
    cov_1st_row[1] = diff_prdjected_noise_freceht_corr[i]
    tau = distance*2/3e8
    m = int(np.floor(tau*sample_rate))
    cov_1st_row[m] += 2*np.pi*linewidth*(tau - (m+1)/sample_rate)
    if (m+1) < len(t):
        cov_1st_row[m+1] += 2*np.pi*linewidth*(m/sample_rate - tau)
    cov = toeplitz(cov_1st_row)
    d_cov_1st_row = np.zeros((len(t)))
    d_cov_1st_row[0] = d_diff_prdjected_noise_freceht_var[i]
    # d_cov_1st_row[1] = -1/2*d_diff_prdjected_noise_freceht_var[i]
    d_cov_1st_row[1] = d_diff_prdjected_noise_freceht_corr[i]
    d_cov_1st_row[m] += 2*np.pi*linewidth*2/3e8
    if (m+1) < len(t):
        d_cov_1st_row[m+1] = -2*np.pi*linewidth*2/3e8
    d_cov = toeplitz(d_cov_1st_row)
    covinv = np.linalg.inv(cov)
    if_func, d_deriv, _ = modf.generate_freq(t, distance, 0, compute_jacobian=True, normalize_freq=True)
    tmp = d_cov@covinv
    # fisher_mat = (2*np.pi)**2 * (d_deriv @ covinv @ d_deriv + 1/2*np.sum(d_cov*covinv) + 1/2*np.sum(tmp*tmp.T))
    # fisher_mat = (2*np.pi)**2 * (d_deriv @ covinv @ d_deriv + np.sum(d_cov*covinv) + 1/2*np.sum(tmp*tmp.T))
    fisher_mat =  d_deriv@covinv@d_deriv * (2*np.pi)**2 # + np.sum(d_cov*covinv) + 1/2*np.sum(tmp*tmp.T)
    d_crb_approx[i] = 1/fisher_mat
    fisher_mat =  fisher_mat + np.sum(d_cov*covinv) + 1/2*np.sum(tmp*tmp.T)
    d_crb_approx_full[i] = 1/fisher_mat
    fisher_mat = ( np.dot(d_deriv,d_deriv) * (2*np.pi)**2 / (4*np.pi*linewidth/sample_rate + diff_prdjected_noise_freceht_var[i]) )**2 
    fisher_mat =  fisher_mat / (d_deriv@cov@d_deriv * (2*np.pi)**2 / (4*np.pi*linewidth/sample_rate + diff_prdjected_noise_freceht_var[i])**2)
    d_crb_approx_iid[i] = 1/fisher_mat
    pbar.update(1)
pbar.close()

np.savez(os.path.join(this_dir, 'd_crb_gaussian_approx_stair600_5_highsnr'), distance=distance_arr, 
         crb_approx=d_crb_approx,
         crb_approx_full=d_crb_approx_full,
         crb_approx_iid=d_crb_approx_iid)

plt.plot(distance_arr, np.sqrt(d_crb_approx_iid), linewidth=3)
plt.legend()
plt.yscale('log')
plt.xlabel('distance (m)')
plt.title('Square root of CRLB')
plt.show()