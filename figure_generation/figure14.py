
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(os.path.join(this_dir, '..', '..'))

import numpy as np
from tqdm import tqdm
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import fmcw_sys
import utils

distance = 500
tau = distance * 2 / 3e8
size_mult = 3

# load parameters
config_filepath = utils.PARAMS_PATH
# config_name = 'tri_2e8_600_5'
config_name = 'sin_2e8_600_5'
meas_prop = fmcw_sys.import_meas_prop_from_config(config_filepath, config_name)
meas_prop.sample_rate /= 10
meas_prop.bandwidth /= 1
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

t = np.arange(0, 2*T, 1/sample_rate)
t = 0.5* (t[1:] + t[:-1])
_, d_deriv0, _ = modf.generate_freq(t, 0, 0, compute_jacobian=True, normalize_freq=True)
outer0 = np.outer(d_deriv0, d_deriv0)

precomputed_data = np.load(os.path.join(this_dir, 'diff_projected_noise_frechet_var_highsnr.npz'))
diff_prdjected_noise_freceht_var_coefs = precomputed_data['frechet_var_polyfit_coefs']
p = np.poly1d(diff_prdjected_noise_freceht_var_coefs)(distance)
diff_prdjected_noise_freceht_corr_coefs = precomputed_data['frechet_corr_polyfit_coefs']
q = np.poly1d(diff_prdjected_noise_freceht_corr_coefs)(distance)
# simulation parameters

def compute_outer_cov(distance):
    t = np.arange(0, 4*T, 1/sample_rate)
    t = 0.5* (t[1:] + t[:-1])
    cov_1st_row = np.zeros((len(t)))
    cov_1st_row[0] = 4*np.pi*linewidth/sample_rate + p
    cov_1st_row[1] = q
    tau = distance*2/3e8
    m = int(np.floor(tau*sample_rate))
    cov_1st_row[m] += 2*np.pi*linewidth*(tau - (m+1)/sample_rate)
    if (m+1) < len(t):
        cov_1st_row[m+1] = 2*np.pi*linewidth*(m/sample_rate - tau)
    cov = toeplitz(cov_1st_row)
    _, d_deriv, _ = modf.generate_freq(t, distance, 0, compute_jacobian=True, normalize_freq=True)
    # d_deriv = t - tau
    outer = np.outer(d_deriv, d_deriv)

    # firstrow = np.zeros((outer.shape[0],))
    # N = len(firstrow)
    # firstrow[0] = 1
    # firstrow[N//6] = 1
    # firstrow[N//6*2] = 1
    # firstrow[N//6*3] = 1
    # firstrow[N//6*4] = 1
    # firstrow[N//6*5] = 1
    # outer = outer + toeplitz(firstrow) * 0.01
    return t, d_deriv, outer, cov

t, d_deriv, outer, cov = compute_outer_cov(distance)
# simulation parameters

# def compute_cov(distance):
#     t = np.arange(0, 6*T, 1/sample_rate)
#     t = 0.5* (t[1:] + t[:-1])
#     cov_1st_row = np.zeros((len(t)))
#     cov_1st_row[0] = 4*np.pi*linewidth/sample_rate
#     tau = distance*2/3e8
#     m = int(np.floor(tau*sample_rate))
#     cov_1st_row[m] += 2*np.pi*linewidth*(tau - (m+1)/sample_rate)
#     if (m+1) < len(t):
#         cov_1st_row[m+1] = 2*np.pi*linewidth*(m/sample_rate - tau)
#     cov = toeplitz(cov_1st_row)
#     # d_deriv = t - tau
#     return cov

# cov = compute_cov(500)

# plt.figure(figsize=(5,5))
# plt.subplot(1,1,1)
# plt.pcolormesh(cov[::-1], vmax=np.amax(cov), vmin=np.amax(cov)*-1, cmap='bwr')


plt.figure(figsize=(8*size_mult,4*size_mult))
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")
fig = plt.gcf()

plt.subplot(1,2,1)
ax = plt.gca()
mesh = plt.pcolormesh(t, t, outer, vmax=np.amax(np.abs(outer)), vmin=-np.amax(np.abs(outer)), cmap='PiYG')

ax.xaxis.tick_top()
ax.invert_yaxis()
ax.set_xticks([0, T, tau, 2*T, 3*T, 4*T-1/sample_rate], ['$0$', '$T$', r'$\tau$', '$2T$', '$3T$', '$4T$'], fontsize=10*size_mult)
ax.set_yticks([0, T, tau, 2*T, 3*T, 4*T-1/sample_rate], ['$0$', '$T$', r'$\tau$', '$2T$', '$3T$', '$4T$'], fontsize=10*size_mult)
ax.set_aspect('equal')

divider = make_axes_locatable(ax)
# below height and pad are in inches
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=10*size_mult)
cbar.ax.set_yticks([-0.01, 0, 0.01])
ax_sub1 = divider.append_axes("top", size='30%', pad=0.5, sharex=ax)
ax_sub2 = divider.append_axes("left", size='30%', pad=0.5, sharey=ax)

ax_sub1.text(T*0.9, 0, '$\\frac{\\partial \\bm g_d}{\\partial d}$', fontsize=15*size_mult)
ax_sub2.text(0.15, T*1.2, '$\\frac{\\partial \\bm g_d^T}{\\partial d}$', fontsize=15*size_mult)

# make some labels invisible

ax_sub1.plot(t, d_deriv, linewidth=4)
ax_sub2.plot(d_deriv, t, linewidth=4)
ax_sub2.invert_xaxis()
ax_sub2.yaxis.tick_right()

ax_sub1.axis('off')
ax_sub2.axis('off')

ax.set_title('Jacobian Outer Product $\\frac{\\partial \\bm g_d^T}{\\partial d} \\frac{\\partial \\bm g_d}{\\partial d}$', y=-0.2, fontsize=15*size_mult)

# plt.setp(ax_sub1.get_xticklabels(), visible=False)
# plt.setp(ax_sub1.get_yticklabels(), visible=False)
# plt.setp(ax_sub2.get_xticklabels(), visible=False)
# plt.setp(ax_sub2.get_yticklabels(), visible=False)
# plt.setp(ax_sub1.get_xticks(), visible=False)
# plt.setp(ax_sub1.get_yticks(), visible=False)
# plt.setp(ax_sub2.get_xticks(), visible=False)
# plt.setp(ax_sub2.get_yticks(), visible=False)
# ax_sub1.set_xticklabels(visible=False)
# ax_sub1.set_yticks([])
# ax_sub2.set_xticks([])
# ax_sub2.set_yticks([])

plt.subplot(1,2,2)
ax = plt.gca()

mesh = plt.pcolormesh(t, t, cov,  vmax=np.amax(cov), vmin=np.amax(cov)*-1, cmap='bwr')

ax.xaxis.tick_top()
ax.invert_yaxis()
ax.set_xticks([0, T, tau, 2*T, 3*T, 4*T-1/sample_rate], ['$0$', '$T$', r'$\tau$', '$2T$', '$3T$', '$4T$'], fontsize=10*size_mult)
ax.set_yticks([0, T, tau, 2*T, 3*T, 4*T-1/sample_rate], ['$0$', '$T$', r'$\tau$', '$2T$', '$3T$', '$4T$'], fontsize=10*size_mult)
ax.set_aspect('equal')
# rect = patches.Rectangle((0, 3+2*len(t)), len(t), len(t), linewidth=2, linestyle='--', edgecolor='black', facecolor='none')
# ax.add_patch(rect)
# rect = patches.Rectangle((1+len(t), 1+len(t)), len(t), len(t), linewidth=2, linestyle='--', edgecolor='black', facecolor='none')
# ax.add_patch(rect)
# rect = patches.Rectangle((3+2*len(t), 0), len(t), len(t), linewidth=2, linestyle='--', edgecolor='black', facecolor='none')
# ax.add_patch(rect)
rect = patches.Rectangle((0, 2*T), 2*T, 2*T, linewidth=2, linestyle='--', edgecolor='black', facecolor='none')
ax.add_patch(rect)
rect = patches.Rectangle((2*T, 0), 2*T, 2*T, linewidth=2, linestyle='--', edgecolor='black', facecolor='none')
ax.add_patch(rect)
# rect = patches.Rectangle((0, 0), 3*len(t)+3, 3*len(t)+3, linewidth=2, linestyle='-', edgecolor='black', facecolor='none')
# ax.add_patch(rect)
rect = patches.Rectangle((0, 0), 4*T, 4*T, linewidth=2, linestyle='-', edgecolor='black', facecolor='none')
ax.add_patch(rect)

# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])

divider = make_axes_locatable(ax)
# below height and pad are in inches
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=10*size_mult)
cbar.ax.set_yticks([-0.06, -0.03, 0, 0.03, 0.06])
ax_sub1 = divider.append_axes("top", size='30%', pad=0.5, sharex=ax)
ax_sub2 = divider.append_axes("left", size='30%', pad=0.5, sharey=ax)
ax_sub1.axis('off')
ax_sub2.axis('off')



ax.set_title('Covariance Matrix $\\bm\\Sigma_d$', y=-0.2, fontsize=15*size_mult)

fig.subplots_adjust(wspace=0.01, left=0.03, right=0.9, bottom=0.15, top=0.95)

plt.show()