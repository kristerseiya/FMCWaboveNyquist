
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))
sys.path.append(os.path.join(this_dir, '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

import fmcw_sys
import utils

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")
size_mult = 3

config_filepath = utils.PARAMS_PATH
config_name = 'sin_2e8_600_5'

meas_prop = fmcw_sys.import_meas_prop_from_config(config_filepath, config_name)
meas_prop.complex_available = True

modf = fmcw_sys.modulation.get_modulation(meas_prop)

T = meas_prop.get_chirp_length()
sample_rate = meas_prop.get_sample_rate()
B = meas_prop.get_bandwidth()

t = np.arange(0, 2*T, 1/sample_rate)

distance_arr = np.arange(0, meas_prop.get_max_d(), 0.5)
velocity_arr = np.arange(-meas_prop.get_max_v(), meas_prop.get_max_v(), 0.5)
if_d2 = np.zeros((len(distance_arr), len(velocity_arr)))
g_hat = modf.generate_freq(t, meas_prop.get_max_d()/2, 0, normalize_freq=True)
for i in np.arange(0, len(distance_arr)):
    for j in np.arange(0, len(velocity_arr)):
        g = modf.generate_freq(t, distance_arr[i], velocity_arr[j], normalize_freq=True)
        if_d2[i, j] = np.sum( (np.mod(g - g_hat + 0.5, 1) - 0.5)**2 )


fig1 = plt.figure(1, figsize=(7*size_mult, 5*size_mult))
fig1.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
ax1 = fig1.add_subplot(111)
mesh = ax1.pcolormesh(distance_arr*2/3e8/2/T, velocity_arr*2/meas_prop.get_carrier_wavelength()/sample_rate, if_d2.T, cmap='Blues')
ax1.set_title("$\\mathcal{D}^2(T,0,\\tau,f)$", fontsize=20*size_mult, y=1.03)
ax1.set_ylabel('normalized doppler shift, $f/f_s$', fontsize=20*size_mult)
ax1.set_xlabel('normalized delay, $\\tau/(2T)$', fontsize=20*size_mult)

ax1.tick_params(axis='both', which='major', labelsize=15*size_mult)

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
cbar=fig1.colorbar(mesh, cax=cax1, orientation='vertical')
cbar.ax.set_yticks([])

d_interval = sample_rate/2/B*T*2*3e8/2
d_interval = np.arcsin(sample_rate/2/B)*T/np.pi*2*3e8/2*2

ax1.arrow((T-d_interval/3e8)/2/T, 0, 2*(d_interval/3e8)/2/T, 0, length_includes_head=True,
width=0.001, color="k", head_width=0.05, head_length=0.05, overhang=0.1, shape='full')
ax1.arrow((T+d_interval/3e8)/2/T, 0, -2*(d_interval/3e8)/2/T, 0, length_includes_head=True,
width=0.001, color="k", head_width=0.05, head_length=0.05, overhang=0.1, shape='full')

ax1.arrow((T)/2/T, -0.5, 0, 1, length_includes_head=True,
width=0.001, color="k", head_width=0.05, head_length=0.05, overhang=0.1, shape='full')
ax1.arrow((T)/2/T, 0.5, 0, -1, length_includes_head=True,
width=0.001, color="k", head_width=0.05, head_length=0.05, overhang=0.1, shape='full')

ax1.text((1.05*T+0.1*d_interval/3e8)/2/T, 0.3, r'$\Delta_f$', fontsize=30*size_mult)
ax1.text((T+d_interval/3e8)/2/T, 0.03, r'$\Delta_\tau$', fontsize=30*size_mult)

ax1.set_ylim(-0.5, 0.5)
ax1.set_xlim(0, T*2/2/T)
plt.show()