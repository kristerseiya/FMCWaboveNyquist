
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.signal
import estimators.grid
import fmcw_sys

import utils

this_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(utils.PROJECT_DIR, 'sim', 'dv_est_sim_results')

results_file = '110125053947_estimates_110125053947_seeds_neval.npz'

plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}

estimator_names = ['wn_snradjust_lattice_faster_11_6',
                   'wn_snradjust_lattice_faster_10_6',
                   'wn_snradjust_lattice_faster_9_6',
                   'wn_snradjust_lattice_faster_8_6',
                   'wn_snradjust_lattice_faster_7_6',
                   'wn_snradjust_lattice_faster_6_6',
                   'wn_snradjust_lattice_faster_5_6',
                   'wn_snradjust_lattice_faster_4_6',
                   'wn_snradjust_lattice_faster_3_6',
                   'wn_snradjust_lattice_faster_11_4',
                   'wn_snradjust_lattice_faster_10_4',
                   'wn_snradjust_lattice_faster_9_4',
                   'wn_snradjust_lattice_faster_8_4',
                   'wn_snradjust_lattice_faster_7_4',
                   'wn_snradjust_lattice_faster_6_4',
                   'wn_snradjust_lattice_faster_5_4',
                   'wn_snradjust_lattice_faster_4_4',
                   'wn_snradjust_lattice_faster_3_4',
                   'wn_snradjust_lattice_faster_11_2',
                   'wn_snradjust_lattice_faster_10_2',
                   'wn_snradjust_lattice_faster_9_2',
                   'wn_snradjust_lattice_faster_8_2',
                   'wn_snradjust_lattice_faster_7_2',
                   'wn_snradjust_lattice_faster_6_2',
                   'wn_snradjust_lattice_faster_5_2',
                   'wn_snradjust_lattice_faster_4_2',
                   'wn_snradjust_lattice_faster_3_2',]

results = np.load(os.path.join(results_dir, results_file))
est_success_rate = np.zeros((9,3))
est_time = np.zeros((9,3))
est_n_evals = np.zeros((9,3))
for estimator_name in estimator_names:
        name_parsed = estimator_name.split("_")
        d_n = int(name_parsed[-2])
        v_n = int(name_parsed[-1])
        i = d_n - 3
        j = v_n//2 - 1
        distance_arr = results['distance']
        velocity_arr = results['velocity']
        xhats = results["estimator_"+estimator_name]
        comp_time = results["estimator_"+estimator_name+"_comp_time"]
        n_evals = results["estimator_"+estimator_name+"_n_evals"]

        dhats = xhats[...,0]
        vhats = xhats[...,1]

        derr = np.abs(np.expand_dims(distance_arr,axis=(1,2))-dhats)
        verr = np.abs(np.expand_dims(velocity_arr,axis=(0,2))-vhats)

        success_rate = np.sum((verr<10)) / derr.size
        
        avg_comp_time = np.mean(comp_time)
        avg_n_evals = np.mean(n_evals)
        
        est_success_rate[i,j] = success_rate
        est_time[i,j] = avg_comp_time
        est_n_evals[i,j] = avg_n_evals

plt.figure(figsize=(13*1.5, 7*1.5))
plt.subplot(121)
mesh = plt.pcolormesh(est_success_rate, vmin=0, vmax=1, cmap="YlOrRd_r")
plt.title("Success Rate", fontsize=25, **font)
ax = plt.gca()
plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5],[3,4,5,6,7,8,9,10,11])
plt.xticks([0.5,1.5,2.5],[2,4,6])
for i in range(est_success_rate.shape[0]):
    for j in range(est_success_rate.shape[1]):
        plt.text(j+0.5, i+0.5, f"{est_success_rate[i,j]:.2f}", ha='center', va='center', fontsize=15)
ax.set_xlabel('Number of Grids in Velocity', fontsize=20, **font)
ax.set_ylabel('Number of Grids in Distance', fontsize=20, **font)
ax.tick_params(axis='both', which='major', labelsize=17)
circle = patches.Circle((0.5, 6.5), 0.4, color='red', alpha=0.7,fill=False, linewidth=2)
ax.add_patch(circle)
fig = plt.gcf()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar=fig.colorbar(mesh, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=2*10)

plt.subplot(122)
mesh = plt.pcolormesh(est_n_evals, cmap="BuPu")
plt.title("Number of Function and Jacobian Evaluations", fontsize=25, **font)
plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5],[3,4,5,6,7,8,9,10,11])
plt.xticks([0.5,1.5,2.5],[2,4,6])
ax = plt.gca()
for i in range(est_n_evals.shape[0]):
    for j in range(est_n_evals.shape[1]):
        plt.text(j+0.5, i+0.5, f"{est_n_evals[i,j]:.2f}", ha='center', va='center', fontsize=15)
ax.set_xlabel('Number of Grids in Velocity', fontsize=20, **font)
ax.set_ylabel('Number of Grids in Distance', fontsize=20, **font)
ax.tick_params(axis='both', which='major', labelsize=17)
circle = patches.Circle((0.5, 6.5), 0.4, color='red', alpha=0.7,fill=False, linewidth=2)
ax.add_patch(circle)
fig = plt.gcf()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar=fig.colorbar(mesh, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=2*10)

plt.show()

