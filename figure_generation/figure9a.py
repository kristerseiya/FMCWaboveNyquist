
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

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")

this_dir = os.path.dirname(os.path.realpath(__file__))

fig, axes = plt.subplots(1,4, figsize=(14,3), width_ratios=[1,5,1,5])
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.3, top=0.7)
ax1 = axes[0]
ax2 = axes[2]

param_name = 'stair_2e8_600_5'
n_cycle = 1
meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join(this_dir, '..', 'fmcw_sys', 'params.json'), 'stair_2e8_600_5')
meas_prop.assume_zero_velocity = True
meas_prop.complex_available = True
meas_prop.modulation = 'triangle'
trif = fmcw_sys.modulation.TriangularModulation(meas_prop)
meas_prop.modulation = 'sinusoidal'
sinf = fmcw_sys.modulation.SinusoidalModulation(meas_prop)
T = meas_prop.get_chirp_length()
sample_rate = meas_prop.get_sample_rate()
B = meas_prop.get_bandwidth()
t = np.arange(0, 2*T, 1/sample_rate)
trifreq = trif.generate_freq_x(t, 0, 0, normalize_freq=True)
sinfreq = sinf.generate_freq_x(t, 0, 0, normalize_freq=True)
ax1.plot(t, trifreq, linewidth=3, color='blue')
l1 = ax2.plot(t, sinfreq, linewidth=3, color='blue')

ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

labelpad = ax1.xaxis.labelpad + 15
for val, letter in [(0, '0'), (2*T, '4\mu s')]:
    ax1.annotate('${}$'.format(letter), xy=(val, 0), xytext=(0, -labelpad),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                ha='center', va='top', size=18)

ax1.annotate('', xy=(0, 0), xytext=(2*T, 0.0),
             xycoords=('data', 'axes fraction'), arrowprops=dict(arrowstyle="<-", color='black'),
             ha='center', va='top', size=18)

ax1.annotate('', xy=(-0.0, 0), xytext=(-0.0, 2.5),
             xycoords=('axes fraction','data'), arrowprops=dict(arrowstyle="<->", color='black'),
             ha='center', va='top', size=18)

ax1.annotate('$B$', xy=(-0.15, 1.25), xytext=(0, 0),
                xycoords=('axes fraction','data'), textcoords='offset points',
                ha='center', va='top', size=18)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1 = ax2

ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

labelpad = ax1.xaxis.labelpad + 15
for val, letter in [(0, '0'),  (2*T, '4\mu s')]:
    ax1.annotate('${}$'.format(letter), xy=(val, 0), xytext=(0, -labelpad),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                ha='center', va='top', size=18)

ax1.annotate('', xy=(0, 0), xytext=(2*T, 0.0),
             xycoords=('data', 'axes fraction'), arrowprops=dict(arrowstyle="<-", color='black'),
             ha='center', va='top', size=18)

ax1.annotate('', xy=(-0.0, 0), xytext=(-0.0, 2.5),
             xycoords=('axes fraction','data'), arrowprops=dict(arrowstyle="<->", color='black'),
             ha='center', va='top', size=18)

ax1.annotate('$B$', xy=(-0.15, 1.25), xytext=(0, 0),
                xycoords=('axes fraction','data'), textcoords='offset points',
                ha='center', va='top', size=18)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1 = axes[1]
ax2 = axes[3]

param_name = 'stair_2e8_600_5'
n_cycle = 1
meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join(this_dir, '..', 'fmcw_sys', 'params.json'), 'stair_2e8_600_5')
meas_prop.assume_zero_velocity = True
meas_prop.complex_available = True
meas_prop.modulation = 'triangle'
trif = fmcw_sys.modulation.TriangularModulation(meas_prop)
meas_prop.modulation = 'sinusoidal'
sinf = fmcw_sys.modulation.SinusoidalModulation(meas_prop)
T = meas_prop.get_chirp_length()
sample_rate = meas_prop.get_sample_rate()
B = meas_prop.get_bandwidth()
t = np.arange(0, 10*T, 1/sample_rate)
trifreq = trif.generate_freq_x(t, 0, 0, normalize_freq=True)
sinfreq = sinf.generate_freq_x(t, 0, 0, normalize_freq=True)
ax1.plot(t, trifreq, linewidth=3, color='purple')
l2 = ax2.plot(t, sinfreq, linewidth=3, color='purple')
    
n_cycle = 1
meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join(this_dir, '..', 'fmcw_sys', 'params.json'), 'tri_2e8_3000_5')
meas_prop.assume_zero_velocity = True
meas_prop.complex_available = True
meas_prop.modulation = 'triangle'
trif = fmcw_sys.modulation.TriangularModulation(meas_prop)
meas_prop.modulation = 'sinusoidal'
sinf = fmcw_sys.modulation.SinusoidalModulation(meas_prop)
T = meas_prop.get_chirp_length()
sample_rate = meas_prop.get_sample_rate()
B = meas_prop.get_bandwidth()
t = np.arange(0, 2*T, 1/sample_rate)
trifreq = trif.generate_freq_x(t, 0, 0, normalize_freq=True)
sinfreq = sinf.generate_freq_x(t, 0, 0, normalize_freq=True)
ax1.plot(t, trifreq, linewidth=3, color='dodgerblue')
l3 = ax2.plot(t, sinfreq, linewidth=3, color='dodgerblue')

ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

labelpad = ax1.xaxis.labelpad + 15
for val, letter in [(0, '0'), (T, '10'), (2*T, '20\mu s')]:
    ax1.annotate('${}$'.format(letter), xy=(val, 0), xytext=(0, -labelpad),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                ha='center', va='top', size=18)

ax1.annotate('', xy=(0, 0), xytext=(2*T, 0.0),
             xycoords=('data', 'axes fraction'), arrowprops=dict(arrowstyle="<-", color='black'),
             ha='center', va='top', size=18)

ax1.annotate('', xy=(-0.0, 0), xytext=(-0.0, 2.5),
             xycoords=('axes fraction','data'), arrowprops=dict(arrowstyle="<->", color='black'),
             ha='center', va='top', size=18)

ax1.annotate('$B$', xy=(-0.05, 1.25), xytext=(0, 0),
                xycoords=('axes fraction','data'), textcoords='offset points',
                ha='center', va='top', size=18)


ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax1 = ax2
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

labelpad = ax1.xaxis.labelpad + 15
for val, letter in [(0, '0'), (T, '10'), (2*T, '20\mu s')]:
    ax1.annotate('${}$'.format(letter), xy=(val, 0), xytext=(0, -labelpad),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                ha='center', va='top', size=18)

ax1.annotate('', xy=(0, 0), xytext=(2*T, 0.0),
             xycoords=('data', 'axes fraction'), arrowprops=dict(arrowstyle="<-", color='black'),
             ha='center', va='top', size=18)

ax1.annotate('', xy=(-0.0, 0), xytext=(-0.0, 2.5),
             xycoords=('axes fraction','data'), arrowprops=dict(arrowstyle="<->", color='black'),
             ha='center', va='top', size=18)

ax1.annotate('$B$', xy=(-0.05, 1.25), xytext=(0, 0),
                xycoords=('axes fraction','data'), textcoords='offset points',
                ha='center', va='top', size=18)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)


fig.legend([l1, l2, l3], 
           labels=[r'$T=2\mu$s, x1', r'$T=2\mu$s, x5', r'$T=10\mu$s, x1'],
           loc=(0.37, 0.72), ncol=3, fontsize=12, framealpha=0)

ax1 = fig.add_axes([0, 0.8, 1, 0.2])
ax1.axis('off')
ax1.annotate('Triangular Modulation', xy=(0.25, 0.5), xytext=(0.25, 0.5),
                xycoords=('axes fraction', 'axes fraction'), textcoords='offset points',
                ha='center', va='center', size=18)
ax1.annotate('Sinusoidal Modulation', xy=(0.75, 0.5), xytext=(0.25, 0.5),
                xycoords=('axes fraction', 'axes fraction'), textcoords='offset points',
                ha='center', va='center', size=18)

plt.show()

