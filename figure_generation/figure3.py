
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
width = 0.05
size_mult = 1

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

fig, axes = plt.subplots(1,2, figsize=(8*size_mult, 6*size_mult))
fig.subplots_adjust(wspace=0.4, hspace=0.5, top=0.88, bottom=0.1, left=0.1, right=0.97)
# ax = axes[0]
# _, = ax.plot(t, modf.generate_freq_x(t, 0, 0), lw=3*size_mult, color='green')
# _, = ax.plot(t, modf.generate_freq_x(t, dist_true[0], vel_true[0]), lw=3*size_mult, color='red')
# ax.set_xlabel('time', fontsize=15*size_mult)
# ax.set_ylabel('frequency', fontsize=15*size_mult)
# ax.set_xticks([0, tau, T, T+tau, 2*T], ['0', r'$\tau$', r'$T$', r'$T+\tau$', r'$2T$'], fontsize=12*size_mult)
# ax.set_yticks([0, B/sample_rate, 0-doppler/sample_rate, B/sample_rate-doppler/sample_rate], 
#               [r'$f_c-\frac{B}{2}$', r'$f_c+\frac{B}{2}$', r'$f_c-\frac{B}{2}-f$', r'$f_c+\frac{B}{2}+f$'], fontsize=12*size_mult)
# ax.vlines([tau, T, T+tau], -0.2, 1, linestyle='--', color='black')
# ax.hlines([0, B/sample_rate, 0-doppler/sample_rate, B/sample_rate-doppler/sample_rate, ], 0, 2*T, linestyle='--', color='black')

# ax.annotate(r'$TX$', xy=(0.7*T, 0.9), xytext=(0, 0),
#                 xycoords=('data','data'), textcoords='offset points',
#                 ha='center', va='top', size=18*size_mult, color='green')

# ax.annotate(r'$RX$', xy=(1.85*T, 0.5), xytext=(0, 0),
#                 xycoords=('data','data'), textcoords='offset points',
#                 ha='center', va='top', size=18*size_mult, color='red')
# ax.set_ylim(-0.2, 1)
# ax.set_xlim(t[0], t[-1])
# ax.set_title('Transmitted vs Received', fontsize=17*size_mult)

ax = axes[0]
_, = ax.plot(t, modf.generate_freq(t, dist_true[0], vel_true[0], aliased=False), lw=3*size_mult, color='blue')

ax.set_xlabel('time', fontsize=15*size_mult)
ax.set_ylabel('frequency', fontsize=15*size_mult)
ax.set_xlim(t[0], t[-1])
ax.set_xticks([0, T, 2*T], ['0', r'$T$', r'$2T$'], fontsize=12*size_mult)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], 
              [r'$-\frac{f_s}{2}$', r'$-\frac{f_s}{4}$', r'$0$', r'$\frac{f_s}{4}$', r'$\frac{f_s}{2}$'], fontsize=12*size_mult)

_, = ax.plot(t, modf.generate_freq(t, dist_true[0]+d_max0/2, vel_true[0], aliased=False), lw=3*size_mult, color='red', linestyle='-')

ax.set_xlabel('time', fontsize=15*size_mult)
ax.set_ylabel('frequency', fontsize=15*size_mult)
ax.set_xlim(t[0], t[-1])
ax.set_xticks([0, T, 2*T], ['0', r'$T$', r'$2T$'], fontsize=12*size_mult)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], 
              [r'$-\frac{f_s}{2}$', r'$-\frac{f_s}{4}$', r'$0$', r'$\frac{f_s}{4}$', r'$\frac{f_s}{2}$'], fontsize=12*size_mult)

_, = ax.plot(t, modf.generate_freq(t, dist_true[0], vel_true[0]-v_max, aliased=False), lw=3*size_mult, color='green', linestyle='-')

ax.set_xlabel('time', fontsize=15*size_mult)
ax.set_ylabel('frequency', fontsize=15*size_mult)
ax.set_ylim(-0.75, 0.75)
ax.set_xlim(t[0], t[-1])
ax.set_xticks([0, T, 2*T], ['0', r'$T$', r'$2T$'], fontsize=12*size_mult)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], 
              [r'$-\frac{f_s}{2}$', r'$-\frac{f_s}{4}$', r'$0$', r'$\frac{f_s}{4}$', r'$\frac{f_s}{2}$'], fontsize=12*size_mult)

ax.annotate(r'$\tau_1, f_1$', xy=(0.6*T, 0.32), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='blue')

ax.annotate(r'$\tau_2, f_1$', xy=(1.75*T, -0.25), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='red')

ax.annotate(r'$\tau_1, f_2$', xy=(0.6*T, -0.18), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='green')

ax.annotate(r'$\tau_2>\tau_1$', xy=(1.6*T, 0.35), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='black')

ax.annotate(r'$f_2<f_1$', xy=(1.6*T, 0.23), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='black')

ax.set_title(r'IF before sampling', fontsize=17*size_mult, y=1.03)

ax = axes[1]

x = modf.generate_freq(t, dist_true[0], vel_true[0], aliased=False)
# _, = ax.plot(t, x, lw=3*size_mult, color='blue')
# _, = ax.plot(t, x+1, lw=3*size_mult, color='blue')
# _, = ax.plot(t, x-1, lw=3*size_mult, color='blue')


xx = np.ma.masked_where(((x>0.5)+(x<-0.5)), x) 
ax.plot(t, xx, lw=3*size_mult, color='blue', linestyle='-')
xx = np.ma.masked_where(((x+1>0.5)+(x+1<-0.5)), x+1) 
ax.plot(t, xx, lw=3*size_mult, color='blue', linestyle='-')
xx = np.ma.masked_where(((x-1>0.5)+(x-1<-0.5)), x-1) 
ax.plot(t, xx, lw=3*size_mult, color='blue', linestyle='-')

xx = np.ma.masked_where(((x<=0.5)*(x>=-0.5)), x) 
ax.plot(t, xx, lw=3*size_mult, color='blue', linestyle=':')
xx = np.ma.masked_where(((x+1<=0.5)*(x+1>=-0.5)), x+1) 
ax.plot(t, xx, lw=3*size_mult, color='blue', linestyle=':')
xx = np.ma.masked_where(((x-1<=0.5)*(x-1>=-0.5)), x-1) 
ax.plot(t, xx, lw=3*size_mult, color='blue', linestyle=':')


ax.set_xlabel('time', fontsize=15*size_mult)
ax.set_ylabel('frequency', fontsize=15*size_mult)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(t[0], t[-1])
ax.set_xticks([0, T, 2*T], ['0', r'$T$', r'$2T$'], fontsize=12*size_mult)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], 
              [r'$-\frac{f_s}{2}$', r'$-\frac{f_s}{4}$', r'$0$', r'$\frac{f_s}{4}$', r'$\frac{f_s}{2}$'], fontsize=12*size_mult)
x = modf.generate_freq(t, dist_true[0]+d_max0/2, vel_true[0], aliased=False)
# _, = ax.plot(t, x, lw=3*size_mult, color='red', linestyle='-')
# _, = ax.plot(t, x+1, lw=3*size_mult, color='red', linestyle='-')
# _, = ax.plot(t, x-1, lw=3*size_mult, color='red', linestyle='-')

xx = np.ma.masked_where(((x>0.5)+(x<-0.5)), x) 
ax.plot(t, xx, lw=3*size_mult, color='red', linestyle='-')
xx = np.ma.masked_where(((x+1>0.5)+(x+1<-0.5)), x+1) 
ax.plot(t, xx, lw=3*size_mult, color='red', linestyle='-')
xx = np.ma.masked_where(((x-1>0.5)+(x-1<-0.5)), x-1) 
ax.plot(t, xx, lw=3*size_mult, color='red', linestyle='-')

xx = np.ma.masked_where(((x<=0.5)*(x>=-0.5)), x) 
ax.plot(t, xx, lw=3*size_mult, color='red', linestyle=':')
xx = np.ma.masked_where(((x+1<=0.5)*(x+1>=-0.5)), x+1) 
ax.plot(t, xx, lw=3*size_mult, color='red', linestyle=':')
xx = np.ma.masked_where(((x-1<=0.5)*(x-1>=-0.5)), x-1) 
ax.plot(t, xx, lw=3*size_mult, color='red', linestyle=':')

ax.set_xlabel('time', fontsize=15*size_mult)
ax.set_ylabel('frequency', fontsize=15*size_mult)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(t[0], t[-1])
ax.set_xticks([0, T, 2*T], ['0', r'$T$', r'$2T$'], fontsize=12*size_mult)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], 
              [r'$-\frac{f_s}{2}$', r'$-\frac{f_s}{4}$', r'$0$', r'$\frac{f_s}{4}$', r'$\frac{f_s}{2}$'], fontsize=12*size_mult)



x = modf.generate_freq(t, dist_true[0], vel_true[0]-v_max, aliased=False)
# _, = ax.plot(t, x, lw=3*size_mult, color='green', linestyle='-')
# _, = ax.plot(t, x+1, lw=3*size_mult, color='green', linestyle='-')
# _, = ax.plot(t, x-1, lw=3*size_mult, color='green', linestyle='-')

xx = np.ma.masked_where(((x>0.5)+(x<-0.5)), x) 
ax.plot(t, xx, lw=3*size_mult, color='green', linestyle='-')
xx = np.ma.masked_where(((x+1>0.5)+(x+1<-0.5)), x+1) 
ax.plot(t, xx, lw=3*size_mult, color='green', linestyle='-')
xx = np.ma.masked_where(((x-1>0.5)+(x-1<-0.5)), x-1) 
ax.plot(t, xx, lw=3*size_mult, color='green', linestyle='-')

xx = np.ma.masked_where(((x<=0.5)*(x>=-0.5)), x) 
ax.plot(t, xx, lw=3*size_mult, color='green', linestyle=':')
xx = np.ma.masked_where(((x+1<=0.5)*(x+1>=-0.5)), x+1) 
ax.plot(t, xx, lw=3*size_mult, color='green', linestyle=':')
xx = np.ma.masked_where(((x-1<=0.5)*(x-1>=-0.5)), x-1) 
ax.plot(t, xx, lw=3*size_mult, color='green', linestyle=':')

ax.set_xlabel('time', fontsize=15*size_mult)
ax.set_ylabel('frequency', fontsize=15*size_mult)
ax.set_ylim(-0.75, 0.75)
ax.set_xlim(t[0], t[-1])
ax.hlines(0.5, t[0], t[-1], lw=5.0 / 2 * size_mult, color='black')
ax.hlines(-0.5, t[0], t[-1], lw=5.0 / 3 * size_mult, color='black')
ax.set_xticks([0, T, 2*T], ['0', r'$T$', r'$2T$'], fontsize=12*size_mult)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], 
              [r'$-\frac{f_s}{2}$', r'$-\frac{f_s}{4}$', r'$0$', r'$\frac{f_s}{4}$', r'$\frac{f_s}{2}$'], fontsize=12*size_mult)

ax.annotate(r'$\tau_1, f_1$', xy=(0.6*T, 0.32), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='blue')

ax.annotate(r'$\tau_2, f_1$', xy=(1.75*T, -0.25), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='red')

ax.annotate(r'$\tau_1, f_2$', xy=(0.6*T, -0.18), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='green')

ax.annotate(r'$\tau_2>\tau_1$', xy=(1.6*T, 0.35), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='black')

ax.annotate(r'$f_2<f_1$', xy=(1.6*T, 0.23), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='top', size=15*size_mult, color='black')
ax.set_title(r'IF after sampling', fontsize=17*size_mult, y=1.03)

plt.show()