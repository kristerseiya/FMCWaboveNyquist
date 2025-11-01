
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal

import utils
import fmcw_sys

this_dir = os.path.dirname(os.path.realpath(__file__))

fig = plt.figure(1, figsize=(9,5))
ax1 = fig.add_subplot(111)
fig.subplots_adjust(left=0.1, right=0.95)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")
size_mult = 3

param_name = 'stair_2e8_600_5'
n_cycle = 1
meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join(this_dir, '..', 'fmcw_sys', 'params.json'), param_name)
meas_prop.assume_zero_velocity = True
meas_prop.complex_available = True
meas_prop.modulation = 'triangle'
trif = fmcw_sys.modulation.TriangularModulation(meas_prop)
meas_prop.modulation = 'sinusoidal'
sinf = fmcw_sys.modulation.SinusoidalModulation(meas_prop)
meas_prop.modulation = 'smoothstairs'
stf= fmcw_sys.modulation.SmoothStairsModulation(meas_prop)
T = meas_prop.get_chirp_length()
sample_rate = meas_prop.get_sample_rate()
B = meas_prop.get_bandwidth()
t = np.arange(0, 2*T, 1/sample_rate)
trifreq = trif.generate_freq_x(t, 0, 0, normalize_freq=True)
sinfreq = sinf.generate_freq_x(t, 0, 0, normalize_freq=True)
stfreq = stf.generate_freq_x(t, 0, 0, normalize_freq=True)
plt.plot(t, trifreq, linewidth=3, color='orange')
plt.plot(t, sinfreq, linewidth=3, color='blue')
plt.plot(t, stfreq, linewidth=3, color='green')
    
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

labelpad = ax1.xaxis.labelpad + 15
for val, letter in [(0, '0'), (T, 'T'), (2*T, '2T')]:
    ax1.annotate('${}$'.format(letter), xy=(val, 0), xytext=(0, -labelpad),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                ha='center', va='top', size=18)

ax1.annotate('', xy=(0, 0), xytext=(2*T, 0.0),
             xycoords=('data', 'axes fraction'), arrowprops=dict(arrowstyle="<-", color='black'),
             ha='center', va='top', size=18)

ax1.annotate('', xy=(-0.0, 0), xytext=(-0.0, 2.5),
             xycoords=('axes fraction','data'), arrowprops=dict(arrowstyle="<->", color='black'),
        #      textcoords='offset points',
             ha='center', va='top', size=18)

ax1.annotate('$B$', xy=(-0.05, 1.25), xytext=(0, 0),
                xycoords=('axes fraction','data'), textcoords='offset points',
                ha='center', va='top', size=18)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1.text(T*0.65, 1.2, 'triangular modulation', **font, fontsize=20)
ax1.text(T*1.3, 2.2, 'sinusoidal modulation', **font, fontsize=20)
ax1.text(T*1.65, 1.4, 'smooth stair \n modulation', **font, fontsize=20)
ax1.plot([T, T*1.2], [1.4, 1.925], color='black')
ax1.plot([T*1.6, T*1.35], [2.1, 1.98], color='black')
ax1.plot([T*1.85, T*1.7], [1.3, 0.97], color='black')
plt.show()

