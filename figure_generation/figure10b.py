
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.signal

import utils

import fmcw_sys

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")

this_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(utils.PROJECT_DIR, 'sim', 'dv_est_sim_results')

fig, axes = plt.subplots(3, 3, figsize=(13*3, 7*3))
fig.subplots_adjust(left=0.15, right=0.95, wspace=0.5, hspace=0.4, bottom=0.1, top=0.85)
ax1 = axes[0,0]
ax2 = axes[1,0]
ax3 = axes[2,0]
ax4 = axes[0,1]
ax5 = axes[1,1]
ax6 = axes[2,1]
ax7 = axes[0,2]
ax8 = axes[1,2]
ax9 = axes[2,2]

mf_filekey = ('102124065331_0_estimates.npz','wn_snradjust_lattice')
lorentz_filekey = ('102124065331_0_estimates.npz','freqavg')
ifreg_filekey = ('100825044516_estimates_100825044516_seeds_mf.npz','matchedfilter2d_optim')

crb_file_path = None
crb_key = None


resfile = lorentz_filekey[0]
results = np.load(os.path.join(results_dir, resfile))
distance_arr = results['distance']
velocity_arr = results['velocity']
dd, vv = np.meshgrid(distance_arr, velocity_arr, indexing='ij')
dv_arr = np.stack([dd,vv],axis=-1)
n_sim = results['n_simulation']
print(n_sim)
max_d = np.amax(distance_arr)

estimator_key = lorentz_filekey[1]
estimate_arr = results['estimator_'+estimator_key]
for i in range(estimate_arr.shape[0]):
        for j in range(estimate_arr.shape[1]):
                for n in range(estimate_arr.shape[2]):
                        if np.all(estimate_arr[i,j,n]==np.array([-111,-111])):
                                estimate_arr[i,j,n] = np.array([max_d, 0])

print(f"{estimator_key:s}, {results['param_name']:s}")

estimate_error = estimate_arr - np.expand_dims(dv_arr, axis=-2)
estimate_rmse = np.sqrt(np.mean((estimate_error)**2,axis=-2))
estimate_rmse_ = (estimate_rmse[:,:,0]<10) * (estimate_rmse[:,:,1]<10)

estimate_rmse[:,:,0] = scipy.signal.convolve(estimate_rmse[:,:,0], np.ones((2,2))/4, mode='same')
estimate_rmse[:,:,1] = scipy.signal.convolve(estimate_rmse[:,:,1], np.ones((2,2))/4, mode='same')
contour3 = ax3.pcolormesh(dd, vv, estimate_rmse_, cmap='Greens', vmin=-0.3, vmax=1.3)

contour2 = ax2.pcolormesh(dd, vv, np.log10(estimate_rmse[:,:,0]), cmap='plasma', vmin=-3.5, vmax=2.5)
contour1 = ax1.pcolormesh(dd, vv, np.log10(estimate_rmse[:,:,1]), cmap='plasma', vmin=-3.5, vmax=2.5)

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
cbar=fig.colorbar(contour1, cax=cax1, orientation='vertical')
cbar.ax.tick_params(labelsize=2*10)
cbar.ax.set_yticks([-3,-2,-1,0,1,2])
cbar.ax.set_yticklabels([1e-3,1e-2,1e-1,1,1e1,1e2])
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
cbar=fig.colorbar(contour2, cax=cax2, orientation='vertical')
cbar.ax.tick_params(labelsize=2*10)
cbar.ax.set_yticks([-3,-2,-1,0,1,2])
cbar.ax.set_yticklabels([1e-3,1e-2,1e-1,1,1e1,1e2])

resfile = ifreg_filekey[0]
results = np.load(os.path.join(results_dir, resfile))
distance_arr = results['distance']
velocity_arr = results['velocity']
dd, vv = np.meshgrid(distance_arr, velocity_arr, indexing='ij')
dv_arr = np.stack([dd,vv],axis=-1)
n_sim = results['n_simulation']
print(n_sim)
max_d = np.amax(distance_arr)

estimator_key = ifreg_filekey[1]
estimate_arr = results['estimator_'+estimator_key]
for i in range(estimate_arr.shape[0]):
        for j in range(estimate_arr.shape[1]):
                for n in range(estimate_arr.shape[2]):
                        if np.all(estimate_arr[i,j,n]==np.array([-111,-111])):
                                estimate_arr[i,j,n] = np.array([max_d, 0])

print(f"{estimator_key:s}, {results['param_name']:s}")

estimate_error = estimate_arr - np.expand_dims(dv_arr, axis=-2)
estimate_rmse = np.sqrt(np.mean((estimate_error)**2,axis=-2))
estimate_rmse_ = (estimate_rmse[:,:,0]<10) * (estimate_rmse[:,:,1]<10)

estimate_rmse[:,:,0] = scipy.signal.convolve(estimate_rmse[:,:,0], np.ones((2,2))/4, mode='same')
estimate_rmse[:,:,1] = scipy.signal.convolve(estimate_rmse[:,:,1], np.ones((2,2))/4, mode='same')
contour6 = ax6.pcolormesh(dd, vv, estimate_rmse_, cmap='Greens', vmin=-0.3, vmax=1.3)

contour4 = ax4.pcolormesh(dd, vv, np.log10(estimate_rmse[:,:,0]), cmap='plasma', vmin=-3.5, vmax=2.5)
contour5 = ax5.pcolormesh(dd, vv, np.log10(estimate_rmse[:,:,1]), cmap='plasma', vmin=-3.5, vmax=2.5)

divider4 = make_axes_locatable(ax7)
cax4 = divider4.append_axes('right', size='5%', pad=0.05)
cbar=fig.colorbar(contour4, cax=cax4, orientation='vertical')
cbar.ax.tick_params(labelsize=2*10)
cbar.ax.set_yticks([-3,-2,-1,0,1,2])
cbar.ax.set_yticklabels([1e-3,1e-2,1e-1,1,1e1,1e2])
divider5 = make_axes_locatable(ax8)
cax5 = divider5.append_axes('right', size='5%', pad=0.05)
cbar=fig.colorbar(contour5, cax=cax5, orientation='vertical')
cbar.ax.tick_params(labelsize=2*10)
cbar.ax.set_yticks([-3,-2,-1,0,1,2])
cbar.ax.set_yticklabels([1e-3,1e-2,1e-1,1,1e1,1e2])


if crb_file_path is not None:
        crb_data = np.load(crb_file_path)
        crb = crb_data[crb_key]
        crb_d = crb_data['distance']
        ax1.plot(crb_d, np.sqrt(crb), linewidth=3, color='red', linestyle='--')

resfile = mf_filekey[0]
results = np.load(os.path.join(results_dir, resfile))
distance_arr = results['distance']
velocity_arr = results['velocity']
dd, vv = np.meshgrid(distance_arr, velocity_arr, indexing='ij')
dv_arr = np.stack([dd,vv],axis=-1)
n_sim = results['n_simulation']
print(n_sim)
max_d = np.amax(distance_arr)

estimator_key = mf_filekey[1]
estimate_arr = results['estimator_'+estimator_key]
for i in range(estimate_arr.shape[0]):
        for j in range(estimate_arr.shape[1]):
                for n in range(estimate_arr.shape[2]):
                        if np.all(estimate_arr[i,j,n]==np.array([-111,-111])):
                                estimate_arr[i,j,n] = np.array([max_d, 0])

print(f"{estimator_key:s}, {results['param_name']:s}")

estimate_error = estimate_arr - np.expand_dims(dv_arr, axis=-2)
estimate_rmse = np.sqrt(np.mean((estimate_error)**2,axis=-2))
estimate_rmse_ = (estimate_rmse[:,:,0]<10) * (estimate_rmse[:,:,1]<10)

estimate_rmse[:,:,0] = scipy.signal.convolve(estimate_rmse[:,:,0], np.ones((2,2))/4, mode='same')
estimate_rmse[:,:,1] = scipy.signal.convolve(estimate_rmse[:,:,1], np.ones((2,2))/4, mode='same')
contour6 = ax9.pcolormesh(dd, vv, estimate_rmse_, cmap='Greens', vmin=-0.3, vmax=1.3)

contour4 = ax7.pcolormesh(dd, vv, np.log10(estimate_rmse[:,:,0]), cmap='plasma', vmin=-3.5, vmax=2.5)
contour5 = ax8.pcolormesh(dd, vv, np.log10(estimate_rmse[:,:,1]), cmap='plasma', vmin=-3.5, vmax=2.5)
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes('right', size='5%', pad=0.05)
cbar=fig.colorbar(contour4, cax=cax4, orientation='vertical')
cbar.ax.tick_params(labelsize=2*10)
cbar.ax.set_yticks([-3,-2,-1,0,1,2])
cbar.ax.set_yticklabels([1e-3,1e-2,1e-1,1,1e1,1e2])
divider5 = make_axes_locatable(ax5)
cax5 = divider5.append_axes('right', size='5%', pad=0.05)
cbar=fig.colorbar(contour5, cax=cax5, orientation='vertical')
cbar.ax.tick_params(labelsize=2*10)
cbar.ax.set_yticks([-3,-2,-1,0,1,2])
cbar.ax.set_yticklabels([1e-3,1e-2,1e-1,1,1e1,1e2])

if crb_file_path is not None:
        crb_data = np.load(crb_file_path)
        crb = crb_data[crb_key]
        crb_d = crb_data['distance']
        ax1.plot(crb_d, np.sqrt(crb), linewidth=3, color='red', linestyle='--')

ax1.set_xlabel('distance (m)', fontsize=12*3, **font)
ax1.set_ylabel('velocity (m/s)', fontsize=12*3, **font)
ax1.tick_params(axis='both', which='major', labelsize=10*3)
ax2.set_xlabel('distance (m)', fontsize=12*3, **font)
ax2.set_ylabel('velocity (m/s)', fontsize=12*3, **font)
ax2.tick_params(axis='both', which='major', labelsize=10*3)
ax3.set_xlabel('distance (m)', fontsize=12*3, **font)
ax3.set_ylabel('velocity (m/s)', fontsize=12*3, **font)
ax3.tick_params(axis='both', which='major', labelsize=10*3)

ax4.set_xlabel('distance (m)', fontsize=12*3, **font)
ax4.set_ylabel('velocity (m/s)', fontsize=12*3, **font)
ax4.tick_params(axis='both', which='major', labelsize=10*3)
ax5.set_xlabel('distance (m)', fontsize=12*3, **font)
ax5.set_ylabel('velocity (m/s)', fontsize=12*3, **font)
ax5.tick_params(axis='both', which='major', labelsize=10*3)
ax6.set_xlabel('distance (m)', fontsize=12*3, **font)
ax6.set_ylabel('velocity (m/s)', fontsize=12*3, **font)
ax6.tick_params(axis='both', which='major', labelsize=10*3)

ax7.set_xlabel('distance (m)', fontsize=12*3, **font)
ax7.set_ylabel('velocity (m/s)', fontsize=12*3, **font)
ax7.tick_params(axis='both', which='major', labelsize=10*3)
ax8.set_xlabel('distance (m)', fontsize=12*3, **font)
ax8.set_ylabel('velocity (m/s)', fontsize=12*3, **font)
ax8.tick_params(axis='both', which='major', labelsize=10*3)
ax9.set_xlabel('distance (m)', fontsize=12*3, **font)
ax9.set_ylabel('velocity (m/s)', fontsize=12*3, **font)
ax9.tick_params(axis='both', which='major', labelsize=10*3)

ax1 = fig.add_axes([0.2, 0.82, 0.1, 0.04])
ax1.axis('off')
ax1.set_title('(i) Tsuchida', fontsize=12*3)

ax1 = fig.add_axes([0.49, 0.82, 0.1, 0.04])
ax1.axis('off')
ax1.set_title('(ii) Matched Filter', fontsize=12*3)

ax1 = fig.add_axes([0.79, 0.82, 0.1, 0.04])
ax1.axis('off')
ax1.set_title('(iii) Proposed', fontsize=12*3)


ax1 = fig.add_axes([0.25+0.05, 0.87, 0.45, 0.04])
ax1.axis('off')
ax1.set_title('(b) Sinusoidal Modulation', fontsize=15*3)


ax = fig.add_axes([0, 0.1, 0.1, 0.9])
ax.axis('off')
ax.text(0.55, 0.755, 'Distance\n RMSE\n (m)', transform=ax.transAxes, fontsize=14*3, va='top', ha='center')
ax.text(0.55, 0.45, 'Velocity\n  RMSE\n (m/s)', transform=ax.transAxes, fontsize=14*3, va='top', ha='center')
ax.text(0.55, 0.14, 'Distance \n RMSE $<$ 10m\n and Velocity \n RMSE $<$ 10m/s', transform=ax.transAxes, fontsize=10*3, va='top', ha='center')

plt.show()

