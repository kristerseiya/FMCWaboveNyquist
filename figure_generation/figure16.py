
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal

import utils
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import copy

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")
size_mult = 3

fig, axes = plt.subplots(1, 3, figsize=(14*size_mult,5*size_mult))

ax1 = axes[0]


this_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(utils.PROJECT_DIR, 'sim', 'd_est_sim_results')


results_filekey = [ ('121224031249_estimates_120924015450_seeds_tri600_highsnr.npz','wn_snradjust_lattice',{'color':'red', 'alpha':0.6, 'label':'Triangular Simulation ($T=2\mu$s, x1)', 'linestyle':'-', 'zorder':3}),
                   ('121224032205_estimates_120924015450_seeds_sin600_highsnr.npz','wn_snradjust_lattice',{'color':'blue', 'alpha':0.6, 'label':'Sinusoidal Simulation ($T=2\mu$s, x1)', 'linestyle':'-', 'zorder':2}),
                    ('121224031826_estimates_120924015450_seeds_stair600_highsnr.npz','wn_snradjust_lattice',{'color':'green', 'alpha':0.6, 'label':'Stairs Simulation ($T=2\mu$s, x1)', 'linestyle':'-', 'zorder':1}),]

crlb_paths = [ ('fisher_approx/d_crb_gaussian_approx_tri600_5_highsnr.npz','crb_approx_iid',{'color':'red', 'label':'Triangular MCRB', 'linestyle':'-', 'linewidth':2*size_mult, 'zorder':6}), 
              #('fisher_approx/d_crb_gaussian_approx_tri600_5_highsnr.npz','crb_approx_full',{'color':'black', 'label':'Triangular MCRB ($T=2\mu$s, x1)', 'linestyle':'--', 'linewidth':4}), 
                ('fisher_approx/d_crb_gaussian_approx_sin600_5_highsnr.npz','crb_approx_iid',{'color':'blue', 'label':'Sinusoidal MCRB', 'linestyle':'-', 'linewidth':2*size_mult, 'zorder':5}),
                ('fisher_approx/d_crb_gaussian_approx_stair600_5_highsnr.npz','crb_approx_iid',{'color':'green', 'label':'Stairs MCRB', 'linestyle':'-', 'linewidth':2*size_mult, 'zorder':4}),
                ]

bin_width = 3

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
                        bin_rmse[k] = np.sqrt(np.mean(rmse[bin_idx==(k+1)]**2))
                        bin_rmse_std[k] = np.sqrt(np.std(rmse[bin_idx==(k+1)]**2))

                window1 = 10
                d_est_rmse_smooth = np.sqrt(scipy.signal.convolve(rmse**2, np.ones((window1,))/window1, mode='same'))
                d_est_rmse_2_smooth = np.sqrt(scipy.signal.convolve(rmse**4, np.ones((window1,))/(window1), mode='same'))
                d_est_rmse_std = (d_est_rmse_2_smooth - window1/(window1)*d_est_rmse_smooth**2)**(1/2)
                ax1.plot(bin_distance_center, bin_rmse, linewidth=2*size_mult, **plot_param)

        crlb_path, crlb_key, plot_params = crlb_paths[i]
        crb_data = np.load(crlb_path)
        mcrb = crb_data["crb_approx_iid"]
        crb = crb_data["crb_approx_full"]
        crb_d = crb_data['distance']
        crb_ub = maximum_filter1d(crb, 3)
        window_size = 7
        crb_ub_ = np.pad(crb_ub, (window_size//2, window_size//2), mode="reflect")
        crb_ub_smooth = scipy.signal.convolve(crb_ub_, np.ones((window_size,))/window_size, mode='valid')
        ax1.plot(crb_d, np.sqrt(mcrb),  **plot_params)
        new_plot_params = copy.deepcopy(plot_params)
        new_plot_params["linestyle"] = "--"
        new_plot_params["label"] = new_plot_params["label"].replace("MCRB", "CRB")
        # ax1.plot(crb_d, np.sqrt(crb_ub_smooth), **new_plot_params)

ax1.set_yscale('log')
ax1.legend(loc='upper left', fontsize=9*size_mult, ncol=1).set_zorder(3*len(results_filekey))
# ax1.legend(loc=(0.23,0.57))
ax1.set_xlabel('target distance (m)', fontsize=15*size_mult)
ax1.set_ylabel('RMSE (m)', fontsize=15*size_mult, **font)
ax1.tick_params(axis='both', which='major', labelsize=10*size_mult)
# ax1.set_title('Sinusoidal Modulation (Observed Length = $10\mu s$)', fontsize=17)
# ax1.set_ylim(2e-4,2e1)
ax1.set_ylim(8e-3,3e-1)
ax1.set_title('(a) Numerical Results and MCRB', fontsize=12*size_mult)
ax1.grid(linewidth=2*size_mult)
ax1.grid(linewidth=1*size_mult, which="minor", linestyle="--")

ax1 = axes[1]

results_filekey = [ ('121224031249_estimates_120924015450_seeds_tri600_highsnr.npz','wn_snradjust_lattice',{'color':'red', 'alpha':0.6, 'label':'Triangular Simulation ($T=2\mu$s, x1)', 'linestyle':'-', 'zorder':3}),
                   ('121224032205_estimates_120924015450_seeds_sin600_highsnr.npz','wn_snradjust_lattice',{'color':'blue', 'alpha':0.6, 'label':'Sinusoidal Simulation ($T=2\mu$s, x1)', 'linestyle':'-', 'zorder':2}),
                    ('121224031826_estimates_120924015450_seeds_stair600_highsnr.npz','wn_snradjust_lattice',{'color':'green', 'alpha':0.6, 'label':'Stairs Simulation ($T=2\mu$s, x1)', 'linestyle':'-', 'zorder':1}),]


crlb_paths = [ ('fisher_approx/d_crb_gaussian_approx_tri600_5_highsnr.npz','crb_approx_iid',{'color':'red', 'label':'Triangular MCRB ($T=2\mu$s, x1)', 'linestyle':'--', 'linewidth':2*size_mult, 'zorder':9}), 
              #('fisher_approx/d_crb_gaussian_approx_tri600_5_highsnr.npz','crb_approx_full',{'color':'black', 'label':'Triangular MCRB ($T=2\mu$s, x1)', 'linestyle':'--', 'linewidth':4}), 
                ('fisher_approx/d_crb_gaussian_approx_sin600_5_highsnr.npz','crb_approx_iid',{'color':'blue', 'label':'Sinusoidal MCRB ($T=2\mu$s, x1)', 'linestyle':'--', 'linewidth':2*size_mult, 'zorder':8}),
                ('fisher_approx/d_crb_gaussian_approx_stair600_5_highsnr.npz','crb_approx_iid',{'color':'green', 'label':'Stairs MCRB ($T=2\mu$s, x1)', 'linestyle':'--', 'linewidth':2*size_mult, 'zorder':7}),
                ]

bin_width = 3

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
                        bin_rmse[k] = np.sqrt(np.mean(rmse[bin_idx==(k+1)]**2))
                        bin_rmse_std[k] = np.sqrt(np.std(rmse[bin_idx==(k+1)]**2))

                window1 = 10
                d_est_rmse_smooth = np.sqrt(scipy.signal.convolve(rmse**2, np.ones((window1,))/window1, mode='same'))
                d_est_rmse_2_smooth = np.sqrt(scipy.signal.convolve(rmse**4, np.ones((window1,))/(window1), mode='same'))
                d_est_rmse_std = (d_est_rmse_2_smooth - window1/(window1)*d_est_rmse_smooth**2)**(1/2)

        crlb_path, crlb_key, plot_params = crlb_paths[i]
        crb_data = np.load(crlb_path)
        mcrb = crb_data["crb_approx_iid"]
        crb = crb_data["crb_approx_full"]
        crb_d = crb_data['distance']
        crb_ub = maximum_filter1d(crb, 3)
        window_size = 7
        crb_ub_ = np.pad(crb_ub, (window_size//2, window_size//2), mode="reflect")
        crb_ub_smooth = scipy.signal.convolve(crb_ub_, np.ones((window_size,))/window_size, mode='valid')
        ax1.plot(crb_d, np.sqrt(mcrb),  **plot_params)
        new_plot_params = copy.deepcopy(plot_params)
        new_plot_params["linestyle"] = "-"
        new_plot_params["zorder"] -= 3
        new_plot_params["label"] = ""
        ax1.plot(crb_d, np.sqrt(crb_ub_smooth), **new_plot_params)
        new_plot_params["alpha"] = 0.6
        new_plot_params["zorder"] -= 3
        new_plot_params["linewidth"] = 0.5*size_mult
        new_plot_params["label"] = plot_params["label"].replace("MCRB", "CRB")
        ax1.plot(crb_d, np.sqrt(crb), **new_plot_params)
        
ax1.set_yscale('log')
ax1.legend(loc='upper left', fontsize=10*size_mult, ncol=1).set_zorder(3*len(results_filekey))
ax1.set_xlabel('target distance (m)', fontsize=15*size_mult)
ax1.set_ylabel('RMSE (m)', fontsize=15*size_mult, **font)
ax1.tick_params(axis='both', which='major', labelsize=10*size_mult)
ax1.set_ylim(8e-3,3e-1)
ax1.set_title('(b) MCRB and CRB', fontsize=12*size_mult)
ax1.grid(linewidth=2*size_mult)
ax1.grid(linewidth=1*size_mult, which="minor", linestyle="--")

ax1 = axes[2]

results_filekey = [ ('121224031249_estimates_120924015450_seeds_tri600_highsnr.npz','wn_snradjust_lattice',{'color':'red', 'alpha':0.6, 'label':'Triangular ($T=2\mu$s, x1)', 'linestyle':'-', 'zorder':2}),
                    ('121224040107_estimates_120924015450_seeds_tri600x5_highsnr.npz','wn_snradjust_lattice',{'color':'orange', 'alpha':0.6, 'label':'Triangular ($T=2\mu$s, x5)', 'linestyle':'-', 'zorder':1}),]

crlb_paths = [ ('fisher_approx/d_crb_gaussian_approx_tri600_5_highsnr.npz','crb_approx_iid',{'color':'red', 'label':'Triangular MCRB ($T=2\mu$s, x1)', 'linestyle':'-', 'linewidth':2*size_mult, 'zorder':5}), 
                ('fisher_approx/d_crb_gaussian_approx_tri600_5_highsnr.npz','crb_approx_iid',{'color':'black', 'label':'Triangular MCRB ($T=2\mu$s, x5 independent)', 'linestyle':'-', 'linewidth':2*size_mult, 'zorder':4}), 
                ('fisher_approx/d_crb_gaussian_approx_tri600_5_x5_highsnr.npz','crb_approx_iid',{'color':'orange', 'label':'Triangular MCRB ($T=2\mu$s, x5 consecutive)', 'linestyle':'-', 'linewidth':2*size_mult, 'zorder':3}), ]

bin_width = 3

for i in range(len(results_filekey)+1):

        if i < 2:
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
                                bin_rmse[k] = np.sqrt(np.mean(rmse[bin_idx==(k+1)]**2))
                                bin_rmse_std[k] = np.sqrt(np.std(rmse[bin_idx==(k+1)]**2))
                        
                        window1 = 10
                        d_est_rmse_smooth = np.sqrt(scipy.signal.convolve(rmse**2, np.ones((window1,))/window1, mode='same'))
                        d_est_rmse_2_smooth = np.sqrt(scipy.signal.convolve(rmse**4, np.ones((window1,))/(window1), mode='same'))
                        d_est_rmse_std = (d_est_rmse_2_smooth - window1/(window1)*d_est_rmse_smooth**2)**(1/2)
                        ax1.plot(bin_distance_center, bin_rmse, linewidth=2*size_mult, **plot_param)

        crlb_path, crlb_key, plot_params = crlb_paths[i]
        crb_data = np.load(crlb_path)
        mcrb = crb_data["crb_approx_iid"]
        if i == 1:
           mcrb = mcrb / 5
        crb = crb_data["crb_approx_full"]
        crb_d = crb_data['distance']
        crb_ub = maximum_filter1d(crb, 3)
        window_size = 7
        crb_ub_ = np.pad(crb_ub, (window_size//2, window_size//2), mode="reflect")
        crb_ub_smooth = scipy.signal.convolve(crb_ub_, np.ones((window_size,))/window_size, mode='valid')
        ax1.plot(crb_d, np.sqrt(mcrb),  **plot_params)
        
ax1.set_yscale('log')
ax1.legend(loc='upper left', fontsize=9*size_mult, ncol=1).set_zorder(3*len(results_filekey))
ax1.set_xlabel('target distance (m)', fontsize=15*size_mult)
ax1.set_ylabel('RMSE (m)', fontsize=15*size_mult, **font)
ax1.tick_params(axis='both', which='major', labelsize=10*size_mult)
ax1.set_ylim(8e-3,3e-1)
ax1.grid(linewidth=2*size_mult)
ax1.grid(linewidth=1*size_mult, which="minor", linestyle="--")


ax1.set_title('(c) Effects of Negative Correlation', fontsize=12*size_mult)

fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15, wspace=0.3)
plt.show()

