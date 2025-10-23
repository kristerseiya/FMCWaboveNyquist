
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.signal
import estimators.grid
import fmcw_sys

import utils

this_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(utils.PROJECT_DIR, 'sim', 'd_est_sim_results')

results_file = '102225002759_estimates_102225002759_seeds_comptime.npz'



# plt.rcParams['text.usetex'] = True
# plt.rcParams["font.family"] = "Times New Roman"
# font = {'fontname':'Times New Roman'}
# plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")

plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}

estimator_names = ['wn_snradjust_lattice_faster', 
                   'lorentz_faster', 'maxpd',
                   'matchedfilter_optim']

results = np.load(os.path.join(results_dir, results_file))
est_time = np.zeros(len(estimator_names))
est_time_std = np.zeros(len(estimator_names))
for esti, estimator_name in enumerate(estimator_names):
    comp_time = results["estimator_"+estimator_name+"_comp_time"]
    avg_comp_time = np.mean(comp_time)
    est_time[esti] = avg_comp_time
    est_time_std[esti] = np.std(comp_time)

print(dict(zip(estimator_names, est_time)))

for estname, esttime, eststd in zip(estimator_names, est_time, est_time_std):
    print(f"{estname:s}, {esttime*1e3:.2f} \u00B1 {eststd*1e3:.2f}")
print()
results_dir = os.path.join(utils.PROJECT_DIR, 'sim', 'dv_est_sim_results')

results_file = '102225003334_estimates_102225003334_seeds_comptime.npz'



# plt.rcParams['text.usetex'] = True
# plt.rcParams["font.family"] = "Times New Roman"
# font = {'fontname':'Times New Roman'}
# plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")

plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}

estimator_names = ['wn_snradjust_lattice_faster', 
                   'lorentz_faster', 'maxpd',
                   'matchedfilter2d_optim']

results = np.load(os.path.join(results_dir, results_file))
est_time = np.zeros(len(estimator_names))
est_time_std = np.zeros(len(estimator_names))
for esti, estimator_name in enumerate(estimator_names):
    comp_time = results["estimator_"+estimator_name+"_comp_time"]
    avg_comp_time = np.mean(comp_time)
    est_time[esti] = avg_comp_time
    est_time_std[esti] = np.std(comp_time)

for estname, esttime, eststd in zip(estimator_names, est_time, est_time_std):
    print(f"{estname:s}, {esttime*1e3:.2f} \u00B1 {eststd*1e3:.2f}")
print()


results_dir = os.path.join(utils.PROJECT_DIR, 'sim', 'd_est_sim_results')

results_file = '102225002249_estimates_102225002249_seeds_comptime.npz'



# plt.rcParams['text.usetex'] = True
# plt.rcParams["font.family"] = "Times New Roman"
# font = {'fontname':'Times New Roman'}
# plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")

plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}

estimator_names = ['wn_snradjust_lattice_faster', 
                   'freqavg',
                   'matchedfilter_optim']

results = np.load(os.path.join(results_dir, results_file))
est_time = np.zeros(len(estimator_names))
est_time_std = np.zeros(len(estimator_names))
for esti, estimator_name in enumerate(estimator_names):
    comp_time = results["estimator_"+estimator_name+"_comp_time"]
    avg_comp_time = np.mean(comp_time)
    est_time[esti] = avg_comp_time
    est_time_std[esti] = np.std(comp_time)

for estname, esttime, eststd in zip(estimator_names, est_time, est_time_std):
    print(f"{estname:s}, {esttime*1e3:.2f} \u00B1 {eststd*1e3:.2f}")
print()

results_dir = os.path.join(utils.PROJECT_DIR, 'sim', 'dv_est_sim_results')


results_file = '102225001542_estimates_102225001542_seeds_comptime.npz'

# plt.rcParams['text.usetex'] = True
# plt.rcParams["font.family"] = "Times New Roman"
# font = {'fontname':'Times New Roman'}
# plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")

plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}

estimator_names = ['wn_snradjust_lattice_faster', 
                   'freqavg',
                   'matchedfilter2d_optim']

results = np.load(os.path.join(results_dir, results_file))
est_time = np.zeros(len(estimator_names))
est_time_std = np.zeros(len(estimator_names))
for esti, estimator_name in enumerate(estimator_names):
    comp_time = results["estimator_"+estimator_name+"_comp_time"]
    avg_comp_time = np.mean(comp_time)
    est_time[esti] = avg_comp_time
    est_time_std[esti] = np.std(comp_time)

for estname, esttime, eststd in zip(estimator_names, est_time, est_time_std):
    print(f"{estname:s}, {esttime*1e3:.2f} \u00B1 {eststd*1e3:.2f}")
print()

