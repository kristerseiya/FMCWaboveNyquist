
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

import fmcw_sys
import time_freq
import estimators

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\usepackage{xcolor}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")
width = 0.07
width2 = 0.1
size_mult = 3

plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 5.0 / 3 * size_mult

# param_name = 'tri_2e8_1000_5'
# param_name = 'tri_2e7_300_2'
# param_name = 'tri_2e8_6000_5'
param_name = 'tri_2e8_600_5'
n_cycle = 1

meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join(this_dir, '..', 'fmcw_sys', 'params.json'), param_name)
meas_prop.assume_zero_velocity = True
meas_prop.complex_available = True
sample_rate = meas_prop.get_sample_rate()
meas_prop.bandwidth = sample_rate/2*2
B = meas_prop.bandwidth
T = meas_prop.Tchirp
d_max0 = sample_rate * T / 2 / B * 3e8 / 2
d_max = meas_prop.max_d
v_max = meas_prop.max_v
lambd_c = meas_prop.lambd_c
print(d_max)
print(v_max)
print(d_max0*2)

# dist_true = np.array([120, 120+d_max0+60])
# vel_true = np.array([20, 20+v_max])
dist_true = np.array([70,])
vel_true = np.array([20,])
# dist_true = np.array([80])
# vel_true = np.array([15])
# dist_true = np.array([80+d_max0])
# vel_true = np.array([15+v_max])
tau_true = 2*dist_true/3e8
stft_window_size = 64
tau = tau_true[0]
doppler = vel_true[0]*2/lambd_c

t = np.arange(0-10/sample_rate, 2*T*n_cycle+10/sample_rate, 1./sample_rate)
fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)

modf = fmcw_sys.modulation.get_modulation(meas_prop)
# modf = fmcw_sys.modulation.SmoothStairsModulation(meas_prop)
x, djac, vjac = modf.generate_freq(t, dist_true, vel_true, compute_jacobian=True)
phase = modf.generate_phase_x(t, dist_true, vel_true)

fig = plt.figure(figsize=(15*size_mult, 5*size_mult))
fig.subplots_adjust(wspace=0.4, top=0.9, bottom=0.15, left=0.1, right=0.97)
spec = fig.add_gridspec(5, 3, height_ratios=[0.15, 0.35, 0.2, 0.35, 0.15])

ax = fig.add_subplot(spec[1:4,0])
_, = ax.plot(t, modf.generate_freq_x(t, 0, 0), lw=3*size_mult, color='green')
_, = ax.plot(t, modf.generate_freq_x(t, dist_true[0], vel_true[0]), lw=3*size_mult, color='red')
ax.set_xlabel('time', fontsize=15*size_mult)
ax.set_ylabel('frequency', fontsize=15*size_mult)
ax.set_xticks([0, tau, T, T+tau, 2*T], ['0', r'$\tau$', r'$T$', r'$T+\tau$', r'$2T$'], fontsize=12*size_mult)
ax.set_yticks([0, B/sample_rate, 0-doppler/sample_rate, B/sample_rate-doppler/sample_rate], 
              [r'$f_c-\frac{B}{2}$', r'$f_c+\frac{B}{2}$', r'$f_c-\frac{B}{2}-f$', r'$f_c+\frac{B}{2}+f$'], fontsize=12*size_mult)
ax.vlines([tau, T, T+tau], -0.2, 1, linestyle='--', color='black')
ax.hlines([0, B/sample_rate, 0-doppler/sample_rate, B/sample_rate-doppler/sample_rate, ], 0, 2*T, linestyle='--', color='black')

ax.annotate(r'$TX$', xy=(0.5*T, 0.75), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=18*size_mult, color='green')

ax.annotate(r'$RX$', xy=(1.75*T, 0.65), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=18*size_mult, color='red')
ax.set_ylim(-0.2, 1)
ax.set_xlim(t[0], t[-1])
ax.set_title('(a) Transmitted vs Received', fontsize=17*size_mult, y=1.03)

ax = fig.add_subplot(spec[0:2,1])
_, = ax.plot(t, modf.generate_freq(t, dist_true[0], vel_true[0], aliased=True), lw=3*size_mult, color='blue')
_, = ax.plot(t, -modf.generate_freq(t, dist_true[0], vel_true[0], aliased=True), lw=3*size_mult, color='blue')
ax.fill_between([tau, T], 
                [B/T*tau/sample_rate+doppler/sample_rate+width, B/T*tau/sample_rate+doppler/sample_rate+width],
                [B/T*tau/sample_rate+doppler/sample_rate-width, B/T*tau/sample_rate+doppler/sample_rate-width],
                #hatch='/', 
                facecolor='#C53B62',
                edgecolor='black')

ax.fill_between([T+tau, 2*T], 
                [B/T*tau/sample_rate-doppler/sample_rate+width, B/T*tau/sample_rate-doppler/sample_rate+width],
                [B/T*tau/sample_rate-doppler/sample_rate-width, B/T*tau/sample_rate-doppler/sample_rate-width],
                #hatch='/', 
                facecolor='#C53B62',
                edgecolor='black')

ax.set_xlabel('time', fontsize=15*size_mult)
ax.set_ylabel('frequency', fontsize=15*size_mult)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(t[0], t[-1])
ax.set_xticks([0, tau, T, T+tau, 2*T], ['0', r'$\tau$', r'$T$', r'$T+\tau$', r'$2T$'], fontsize=12*size_mult)
# ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], 
#               [r'$\frac{f_s}{2}$', r'$\frac{f_s}{4}$', r'$0$', r'$\frac{f_s}{4}$', r'$\frac{f_s}{2}$'], fontsize=12*size_mult)
ax.set_yticks([-0.5, 0, 0.5], 
              [r'$-\frac{f_s}{2}$',  r'$0$',  r'$\frac{f_s}{2}$'], fontsize=12*size_mult)

ax.vlines([tau, T, T+tau], -0.5, 0.5, linestyle='--', color='black')

ax.annotate(r'$\frac{B}{T}\tau+f$', xy=(0.60*T, B/T*tau/sample_rate+doppler/sample_rate-0.2), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='center', size=12*size_mult)

ax.annotate(r'$\frac{B}{T}\tau-f$', xy=(1.62*T, B/T*tau/sample_rate-doppler/sample_rate+0.2), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='center', size=12*size_mult)
ax.set_title('(b) Real Signal', fontsize=17*size_mult, y=1.03)

ax = fig.add_subplot(spec[3:5,1])
_, = ax.plot(t, modf.generate_freq(t, dist_true[0], vel_true[0], aliased=True), lw=3*size_mult, color='blue')

x = modf.generate_freq(t+20/sample_rate, dist_true[0], vel_true[0], aliased=True)
y = modf.generate_freq(t-20/sample_rate, dist_true[0], vel_true[0], aliased=True)
z1 = np.maximum(x+width2, y+width2)
z2 = np.minimum(x-width2, y-width2)
ax.fill_between(t,
                z1, z2,
                #hatch='/', 
                facecolor='#9DC3E6',
                edgecolor='black')

ax.fill_between([tau, T], 
                [B/T*tau/sample_rate+doppler/sample_rate+width, B/T*tau/sample_rate+doppler/sample_rate+width],
                [B/T*tau/sample_rate+doppler/sample_rate-width, B/T*tau/sample_rate+doppler/sample_rate-width],
                #hatch='/', 
                facecolor='#FFC000',
                edgecolor='black')

ax.fill_between([T+tau, 2*T], 
                [-B/T*tau/sample_rate+doppler/sample_rate+width, -B/T*tau/sample_rate+doppler/sample_rate+width],
                [-B/T*tau/sample_rate+doppler/sample_rate-width, -B/T*tau/sample_rate+doppler/sample_rate-width],
                #hatch='/', 
                facecolor='#FFC000',
                edgecolor='black')

ax.set_xlabel('time', fontsize=15*size_mult)
ax.set_ylabel('frequency', fontsize=15*size_mult)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(t[0], t[-1])
ax.set_xticks([0, tau, T, T+tau, 2*T], ['0', r'$\tau$', r'$T$', r'$T+\tau$', r'$2T$'], fontsize=12*size_mult)
# ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], 
#               [r'$\frac{f_s}{2}$', r'$\frac{f_s}{4}$', r'$0$', r'$\frac{f_s}{4}$', r'$\frac{f_s}{2}$'], fontsize=12*size_mult)
ax.set_yticks([-0.5, 0, 0.5], 
              [r'$-\frac{f_s}{2}$',  r'$0$',  r'$\frac{f_s}{2}$'], fontsize=12*size_mult)


ax.vlines([tau, T, T+tau], -0.5, 0.5, linestyle='--', color='black')

ax.annotate(r'$\frac{B}{T}\tau+f$', xy=(0.60*T, B/T*tau/sample_rate+doppler/sample_rate-0.15), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=12*size_mult)

ax.annotate(r'$-\frac{B}{T}\tau+f$', xy=(1.62*T, -B/T*tau/sample_rate+doppler/sample_rate+0.3), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=12*size_mult)

ax.set_title('(c) Complex Signal', fontsize=17*size_mult, y=1.03)

ax = fig.add_subplot(spec[1:4,2])

rect = Rectangle((0, -sample_rate/2), 2*T, sample_rate, edgecolor='black', lw=2*size_mult, facecolor='#9DC3E6')
ax.add_patch(rect)

rect = Rectangle((0, -sample_rate/4), sample_rate/B*T, sample_rate/2, edgecolor='black', lw=2*size_mult,facecolor='#FFC000')
ax.add_patch(rect)

ax.fill_between([0, sample_rate*T/B/4, sample_rate*T/B/2],
                [0, sample_rate/4, 0],
                [0, -sample_rate/4, 0], edgecolor='black', facecolor='#C53B62', lw=2*size_mult)

ax.set_xticks([0, sample_rate*T/B/2, sample_rate*T/B, 2*T], 
              [r'$0$', r'$\frac{Tf_s}{2B}$', r'$\frac{Tf_s}{B}$', r'$2T$', ], fontsize=12*size_mult)
ax.set_yticks([-sample_rate/2, -sample_rate/4, 0, sample_rate/4, sample_rate/2], 
              [r'$-\frac{f_s}{2}$', r'$-\frac{f_s}{4}$', '$0$', r'$\frac{f_s}{4}$', r'$\frac{f_s}{2}$'], fontsize=12*size_mult)

ax.annotate('CBF\n(Real)', xy=(sample_rate*T/B/4, 0), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='center', size=10*size_mult)

ax.annotate('CBF\n(Complex)', xy=(sample_rate*T/B/4*3, 0), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='center', size=10*size_mult)

x = 0.5 * (2*T + sample_rate*T/B)
ax.annotate('Proposed\n(Complex)', xy=(x, 0), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='center', size=10*size_mult)

ax.set_title('(d) Unambiguous Space', fontsize=17*size_mult, y=1.03)
ax.set_xlabel(r'delay, $\tau$', fontsize=15*size_mult)
ax.set_ylabel(r'Doppler, $f$', fontsize=15*size_mult)
ax.set_xlim(0, 2*T)
ax.set_ylim(-sample_rate/2, sample_rate/2)


plt.show()