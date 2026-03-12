
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, iv
from scipy.fftpack import fft, ifft
from matplotlib.widgets import Slider

from scipy.interpolate import make_smoothing_spline
import fmcw_sys

np.random.seed(125151)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath \usepackage{amsfonts} \usepackage{amsmath} \renewcommand{\seriesdefault}{\bfdefault}")

mu1 = 1
mu2 = 0
A = np.sqrt(mu1**2 + mu2**2)

n_sample = 10000

test_snr = np.linspace(-20, 30, 10000)
test_sigma2 = 10**(-test_snr/10) * A**2
test_var = np.zeros_like(test_sigma2)
test_corr = np.zeros_like(test_sigma2)

for i, sigma2 in enumerate(test_sigma2):
    signal = mu1 + 1j * mu2 + (np.random.randn(n_sample,)*np.sqrt(0.5*sigma2) + 1j*np.random.randn(n_sample,)*np.sqrt(0.5*sigma2))
    signal = signal[1:] * np.conj(signal[:-1])
    phase = np.angle(signal) / (2*np.pi)
    test_var[i] = np.var(phase)
    test_corr[i] = np.mean(phase[:-1]*phase[1:])

spl = make_smoothing_spline(test_snr, test_var, lam=1)
spl2 = make_smoothing_spline(test_snr, test_corr, lam=1)

polyfit = np.polyfit(test_snr, test_var, 20)
polyfit2 = np.polyfit(test_snr, test_corr, 20)

asymptote1 = np.ones_like(test_snr)*1/12
asymptote2 = 10**(-test_snr/10)/(2*np.pi)**2

fig = plt.figure(figsize=(13,6))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.3)
ax = fig.add_subplot(121)
test_var_line, = ax.plot(test_snr, test_var, linewidth=3, label=r'$\mathbb{E}[\epsilon_n^2]$ simulated')
ax.plot(test_snr, spl(test_snr), label=r'$\mathbb{E}[\epsilon_n^2]$ fitted')
ax.set_xlabel(r"$10\log_{10}\textit{SNR}_\eta (dB)$", fontsize=25)
ax.set_ylabel(r"$\text{var}(\epsilon_n)$", fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(fontsize=20, loc='upper right')
ax.grid()
ax.set_ylim(-1/72, 1/12+1/72)
ax = fig.add_subplot(122)
ax.plot(test_snr, np.log10(asymptote1), linewidth=2, linestyle='--', color='red')
ax.plot(test_snr, np.log10(asymptote2), linewidth=2, linestyle='--', color='purple')
test_var_line, = ax.plot(test_snr, np.log10(test_var), linewidth=3, label=r'$\mathbb{E}[\epsilon_n^2]$ simulated')
ax.plot(test_snr, np.log10(spl(test_snr)), label=r'$\mathbb{E}[\epsilon_n^2]$ fitted')
ax.annotate(r'$\text{var}(\epsilon_n)=\frac{1}{12}$', xy=(15, -1.5), xytext=(0, 0), color='red',
                xycoords=('data','data'), textcoords='offset points',
                ha='center', va='center', size=25)
ax.annotate(r'$\text{var}(\epsilon_n)=\frac{1}{(2\pi)^2 \textit{SNR}_\eta}$', xy=(0, -3.5), xytext=(0, 0),
                xycoords=('data','data'), textcoords='offset points', color='purple',
                ha='center', va='center', size=25)

ax.set_xlabel(r"$10\log_{10}\textit{SNR}_\eta (dB)$", fontsize=25)
ax.set_ylabel(r"$\log_{10}\text{var}(\epsilon_n)$", fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(fontsize=20, loc='upper right')
ax.grid()
fig.suptitle(r'$\hat{h}(\textit{SNR}_{\eta})$', fontsize=27)
ax.set_ylim(np.log10(1/72)-3, np.log10(1/72)+2.5)


fig = plt.figure(figsize=(13,6))
fig.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.90)
ax = fig.add_subplot(121)
test_var_line, = ax.plot(test_snr, test_var, linewidth=3, label=r'$\mathbb{E}[\epsilon_n^2]$ simulated')
ax.plot(test_snr, spl(test_snr), label=r'$\mathbb{E}[\epsilon_n^2]$ fitted')
ax.plot(test_snr, test_corr, linewidth=3, label=r'$\mathbb{E}[\epsilon_{n+1}\epsilon_n]]$ simulated')
ax.plot(test_snr, spl2(test_snr), label=r'$\mathbb{E}[\epsilon_{n+1}\epsilon_n]]$ fitted')
ax.set_xlabel(r"$10\log_{10}\textit{SNR}_\eta (dB)$", fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('Variance / Correlation', fontsize=27)
ax.legend(fontsize=17, loc='upper right')
ax.grid()
ax = fig.add_subplot(122)
ax.plot(test_snr, test_corr/test_var, color='orange', label='simulated')
ax.plot(test_snr, spl2(test_snr)/spl(test_snr), label='fitted')
ax.set_xlabel(r"$10\log_{10}\textit{SNR}_\epsilon (dB)$", fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('Correlation Coefficient', fontsize=27)
ax.grid()
ax.legend(fontsize=20, loc='upper right')

param_name = 'sin_2e8_600_5'
n_cycle = 1

meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join(this_dir, '..', 'fmcw_sys', 'params.json'), param_name)
meas_prop.assume_zero_velocity = True
meas_prop.complex_available = True

Rpd = meas_prop.detector_effectivity
q = 1.6e-19
Ptx = meas_prop.transmitted_power
R = meas_prop.reflectance
A = meas_prop.detector_effective_area
Plo = Ptx
distance = 600
Prx = Ptx * A / (np.pi*distance**2) * R
snr = 10 * np.log10( 2 * Rpd**2 * Plo * Prx ) - 10*np.log10( q*Rpd*(Plo+Prx) )
plt.subplot(121)
ax = plt.gca()
ax.vlines(snr, ymin=0.65/(2*np.pi)**2, ymax=0.85/(2*np.pi)**2, color='red', linewidth=2)
ax.text(snr+4, 1.0/(2*np.pi)**2, s='operating\n region', fontsize=20, ha='center', color='red')
ax.plot([snr, 30], [0.75/(2*np.pi)**2, 0.75/(2*np.pi)**2], color='red', linewidth=2)

plt.show()

# np.save(os.path.join(this_dir,'project_normal_diff_var_polyfit'), polyfit)