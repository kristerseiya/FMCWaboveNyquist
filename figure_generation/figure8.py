
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(os.path.join(this_dir, '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

import utils
import time_freq
import estimators
import fmcw_sys

np.random.seed(1215120)

size_mult = 3

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")

# param_name = 'tri_2e8_1000_5_lw1e5'
param_name = 'tri_2e8_600_5'
# param_name = 'sin_2e8_1000_5_lw1e5'
meas_prop = fmcw_sys.import_meas_prop_from_config(utils.PARAMS_PATH, param_name)
meas_prop.assume_zero_velocity = False
meas_prop.complex_available = True
# meas_prop.linewidth = 1e6

dist_true = 300
vel_true = 0
tau_true = 2*dist_true/3e8

sample_rate = meas_prop.get_sample_rate()
T = meas_prop.get_chirp_length()
t = np.arange(0, 2*T, 1./sample_rate)
fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
signal, second_output = fmcw_meas.generate(dist_true, vel_true, t)

vmax = meas_prop.get_max_v()
dmax = meas_prop.get_max_d()

# estimator = estimators.get_estimators(meas_prop, ['snradjust_uniformgrid'])[0]
# estimator = estimators.get_estimators(meas_prop, ['shortestpath_uniformgrid'])[0]
key = {'gridgen_type':'optimal', 'method':'gradient_descent', 'init_step': 'none', 'ignore_quadrature':False, 'snr_adjustment':True, 'gd_max_n_iter':500}
estimator = estimators.IFRegressor(meas_prop, **key, average=True)

estimator.set_test_flag(True)
x_hat, intermediate_results = estimator.estimate(t, signal, second_output)
d_arr, v_arr = np.linspace(0.1, dmax, 100), np.linspace(-vmax, vmax, 100)
grid_d, grid_v = np.meshgrid(d_arr, v_arr)

dvpair = np.stack([grid_d, grid_v], axis=-1)
dvpair = dvpair.reshape((-1, 2))

print('    True: {:.2f}m, {:.2f}m/s'.format(dist_true, vel_true))
print('Estimate: {:.2f}m, {:.2f}m/s'.format(x_hat[0], x_hat[1]))

print(intermediate_results.keys())
x_init = intermediate_results['x_init']
x_hats = intermediate_results['x_hat']
gdtracks = intermediate_results['gd_track']
model = intermediate_results['likelihood_model']
loss1 = model.evaluate(dvpair, extra_var=0.05)
loss2 = model.evaluate(dvpair, extra_var=0.0008)
loss3 = model.evaluate(dvpair, extra_var=0)

loss1 = loss1.reshape(grid_d.shape)
loss2 = loss2.reshape(grid_d.shape)
loss3 = loss3.reshape(grid_d.shape)


fig1 = plt.figure(1, figsize=(14*size_mult, 4*size_mult))
fig1.subplots_adjust(wspace=0.3, left=0.05, right=0.99, bottom=0.15, top=0.9)
ax1 = fig1.add_subplot(141)
ax2 = fig1.add_subplot(142)
ax3 = fig1.add_subplot(143)
ax4 = fig1.add_subplot(144)

ax1.contourf(d_arr, v_arr, loss1)
ax2.contourf(d_arr, v_arr, loss1)
ax3.contourf(d_arr, v_arr, loss2)
ax4.contourf(d_arr, v_arr, loss3)


print(gdtracks.shape)
idx_part = [[0, 10], [30, 50], [50, -1]]
for gdtrack in gdtracks:
    x, y = gdtrack[0, idx_part[0][0]:idx_part[0][1]], gdtrack[1, idx_part[0][0]:idx_part[0][1]]
    ax2.plot(x, y, color='magenta', zorder=1, linewidth=size_mult*3)
    ax2.plot(x, y-2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax2.plot(x, y+2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax2.plot(x-dmax, y, color='magenta', zorder=1, linewidth=size_mult*3)
    ax2.plot(x-dmax, y-2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax2.plot(x-dmax, y+2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax2.plot(x+dmax, y, color='magenta', zorder=1, linewidth=size_mult*3)
    ax2.plot(x+dmax, y-2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax2.plot(x+dmax, y+2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)

    x, y = gdtrack[0, idx_part[1][0]:idx_part[1][1]], gdtrack[1, idx_part[1][0]:idx_part[1][1]]
    ax3.plot(x, y, color='magenta', zorder=1, linewidth=size_mult*3)
    ax3.plot(x, y-2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax3.plot(x, y+2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax3.plot(x-dmax, y, color='magenta', zorder=1, linewidth=size_mult*3)
    ax3.plot(x-dmax, y-2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax3.plot(x-dmax, y+2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax3.plot(x+dmax, y, color='magenta', zorder=1, linewidth=size_mult*3)
    ax3.plot(x+dmax, y-2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax3.plot(x+dmax, y+2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)

    x, y = gdtrack[0, idx_part[2][0]:idx_part[2][1]], gdtrack[1, idx_part[2][0]:idx_part[2][1]]
    ax4.plot(x, y, color='magenta', zorder=1, linewidth=size_mult*3)
    ax4.plot(x, y-2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax4.plot(x, y+2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax4.plot(x-dmax, y, color='magenta', zorder=1, linewidth=size_mult*3)
    ax4.plot(x-dmax, y-2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax4.plot(x-dmax, y+2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax4.plot(x+dmax, y, color='magenta', zorder=1, linewidth=size_mult*3)
    ax4.plot(x+dmax, y-2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)
    ax4.plot(x+dmax, y+2*vmax, color='magenta', zorder=1, linewidth=size_mult*3)


ax1.scatter(x_init[:,0], x_init[:,1], color='red', zorder=2, label='initial', s=30*size_mult)

# ax2.scatter(x_init[:,0], x_init[:,1], color='red', zorder=2, label='initial', s=30)
# ax2.scatter(x_hats[:,0], x_hats[:,1], color='blue', zorder=3, label='converged', s=20)
ax2.scatter(gdtracks[:, 0, idx_part[0][0]], gdtracks[:, 1, idx_part[0][0]], color='red', zorder=2, label='start', s=30*size_mult**2)
ax2.scatter(gdtracks[:, 0, idx_part[0][1]], gdtracks[:, 1, idx_part[0][1]], color='mediumpurple', zorder=3, label='end', s=20*size_mult**2)

# ax2.scatter([x_hat[0],], [x_hat[1],], color='orange', zorder=4, label='optimal')

ax3.scatter(gdtracks[:, 0, idx_part[1][0]], gdtracks[:, 1, idx_part[1][0]], color='mediumpurple', zorder=2, label='start', s=30*size_mult**2)
ax3.scatter(gdtracks[:, 0, idx_part[1][1]], gdtracks[:, 1, idx_part[1][1]], color='mediumturquoise', zorder=3, label='end', s=20*size_mult**2)

ax4.scatter(gdtracks[:, 0, idx_part[2][0]], gdtracks[:, 1, idx_part[2][0]], color='mediumturquoise', zorder=2, label='start', s=30*size_mult**2)
ax4.scatter(gdtracks[:, 0, idx_part[2][1]], gdtracks[:, 1, idx_part[2][1]], color='blue', zorder=3, label='end', s=20*size_mult**2)

ax4.scatter([x_hat[0],], [x_hat[1],], color='orange', zorder=4, label='optimal', s=30*size_mult**2)


ax1.tick_params(axis='both', which='major', labelsize=10*size_mult)
ax2.tick_params(axis='both', which='major', labelsize=10*size_mult)
ax3.tick_params(axis='both', which='major', labelsize=10*size_mult)
ax4.tick_params(axis='both', which='major', labelsize=10*size_mult)

ax1.set_xlim(0, dmax)
ax1.set_ylim(-vmax, vmax)
ax2.set_xlim(0, dmax)
ax2.set_ylim(-vmax, vmax)
ax1.legend(loc='upper left', fontsize=10*size_mult)
ax2.legend(loc='upper left', fontsize=10*size_mult)

ax3.set_xlim(0, dmax)
ax3.set_ylim(-vmax, vmax)
ax4.set_xlim(0, dmax)
ax4.set_ylim(-vmax, vmax)
ax3.legend(loc='upper left', fontsize=10*size_mult)
ax4.legend(loc='upper left', fontsize=10*size_mult)

ax1.set_title('Initialization', fontsize=15*size_mult)
ax2.set_title('Gradient Descent $\sigma_1$', fontsize=15*size_mult)
ax1.set_ylabel('velocity (m/s)', fontsize=15*size_mult)
ax1.set_xlabel('distance (m)', fontsize=15*size_mult)
ax2.set_xlabel('distance (m)', fontsize=15*size_mult)
ax2.set_ylabel('velocity (m/s)', fontsize=15*size_mult)

ax3.set_title('Gradient Descent $\sigma_2$', fontsize=15*size_mult)
ax4.set_title('Gradient Descent $\sigma_3$', fontsize=15*size_mult)
ax3.set_ylabel('velocity (m/s)', fontsize=15*size_mult)
ax3.set_xlabel('distance (m)', fontsize=15*size_mult)
ax4.set_xlabel('distance (m)', fontsize=15*size_mult)
ax4.set_ylabel('velocity (m/s)', fontsize=15*size_mult)
# fig2 = plt.figure(2)
# ax21 = fig2.add_subplot(111)
# # ax21.contourf(uniform_grid2d[:,:,0], uniform_grid2d[:,:,1], grid_vals)

plt.show()
