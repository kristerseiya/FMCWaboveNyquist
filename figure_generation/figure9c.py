
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal

import utils

this_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(utils.PROJECT_DIR, 'sim', 'd_est_sim_results')

results_filekey = [('100725084519_estimates_100725084519_seeds_tri3000_moresim.npz','wn_snradjust_lattice_faster',{'color':'dodgerblue', 'label':'Proposed ($T=10\mu s$, x1)'}), 
                   ('101425040042_estimates_100725084519_seeds_lorentz_redo.npz','lorentz',{'color':'orange', 'label':'Lorentzian Regression ($T=10\mu s$, x1)'}),
                ('101325192646_estimates_100725084519_seeds_maxpd_redo.npz','maxpd',{'color':'green', 'label':'Maximum Periodogram ($T=10\mu s$, x1)'}),
                ('100825111525_estimates_100725084519_seeds_tri3000mf.npz','matchedfilter_optim',{'color':'grey', 'label':'Matched Filter ($T=10\mu s$, x1)'}),
                ('100725073440_estimates_100725073440_seeds_tri600_5_moresim.npz','wn_snradjust_lattice_faster',{'color':'purple', 'label':'Proposed ($T=2\mu s$, x5)'}),
                ('100825110447_estimates_100725073440_seeds_tri600_5mf.npz','matchedfilter_optim',{'color':'darkgoldenrod', 'label':'Matched Filter ($T=2\mu s$, x5)'}),  
                ]

plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}

fig = plt.figure(1, figsize=(9,12))
ax1 = fig.add_subplot(111)
fig.subplots_adjust(left=0.1, right=0.95)

bin_width = 1

for i in range(len(results_filekey)):

        resfile = results_filekey[i][0]
        results = np.load(os.path.join(results_dir, resfile))
        distance_arr = results['distance']
        n_sim = results['n_simulation']
        visible_idx = distance_arr <= 595
        distance_arr = distance_arr[visible_idx]
        max_d = np.amax(distance_arr)
        min_d = np.amin(distance_arr)
        bin_distance_edges = np.arange(min_d, max_d+bin_width, bin_width)
        bin_distance_center = 0.5*(bin_distance_edges[1:]+bin_distance_edges[:-1])

        estimator_keys = results_filekey[i][1]
        if estimator_keys == 'all': 
                estimator_keys = list()
                for key in results.keys():
                        if key.startswith('estimator_'):
                                estimator_keys.append(key.removeprefix('estimator_'))
        else:
                estimator_keys = [estimator_keys]

        print(results['param_name'])
        print(results['n_simulation'])
        print(estimator_keys)

        if len(results_filekey[i]) == 3:
                plot_param = results_filekey[i][2]
        else:
                plot_param = dict()
        for estimator_key in estimator_keys:
                estimate_arr = results['estimator_'+estimator_key]
                estimate_arr = estimate_arr[visible_idx]
                estimate_arr[estimate_arr==-111] = max_d
                rmse = np.sqrt(np.mean((estimate_arr.T-distance_arr)**2,axis=0))
                bin_rmse = np.zeros_like(bin_distance_center)
                bin_rmse_std = np.zeros_like(bin_distance_center)
                bin_idx = np.digitize(distance_arr, bin_distance_edges)
                for k in range(len(bin_distance_center)):
                        bin_rmse[k] = np.mean(rmse[bin_idx==(k+1)])
                        bin_rmse_std[k] = np.std(rmse[bin_idx==(k+1)])
                window1 = 10
                d_est_rmse_smooth = scipy.signal.convolve(rmse, np.ones((window1,))/window1, mode='same')
                d_est_rmse_2_smooth = scipy.signal.convolve(rmse**2, np.ones((window1,))/(window1), mode='same')
                d_est_rmse_std = (d_est_rmse_2_smooth - window1/(window1)*d_est_rmse_smooth**2)**(1/2)
                ax1.plot(bin_distance_center, bin_rmse, linewidth=2.5, **plot_param, zorder=(len(results_filekey)*2-i))
                ax1.fill_between(bin_distance_center, bin_rmse-bin_rmse_std, bin_rmse+bin_rmse_std, alpha=0.4, color=plot_param['color'], zorder=(len(results_filekey)-i))

ax1.set_yscale('log')
ax1.legend(loc='upper left', fontsize=25).set_zorder(3*len(results_filekey))
ax1.set_xlabel('target distance (m)', fontsize=35)
ax1.set_ylabel('RMSE (m)', fontsize=35)
ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.set_ylim(8e-4,8e4)
ax1.grid(linewidth=2)
fig.tight_layout()
plt.show()

