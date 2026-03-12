
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, rfft, irfft
import utils
import os

import fmcw_sys
from scipy.fftpack import fft, ifft
from matplotlib import cm


param_name = 'tri_2e8_600_5'
n_cycle = 1
L = 1e5

meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join('fmcw_sys', 'params.json'), param_name)
meas_prop.linewidth = L
dist_true = 300
vel_true = 0
tau_true = 2*dist_true/3e8
stft_window_size = 64

np.random.seed(276358)
meas_prop.assume_zero_velocity = False
sample_rate = meas_prop.get_sample_rate()
T = meas_prop.get_chirp_length()
t = np.arange(0, 2*T, 1./sample_rate)
fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
signal, second_output = fmcw_meas.generate(dist_true, vel_true, t, include_shot_noise=True)

phase = fmcw_meas.generate_phase(t, 0, 0)
ref = np.exp(1j*phase)

ref_h_t = np.arange(0, 2*T, 1./sample_rate/6)
phase = fmcw_meas.generate_phase(ref_h_t, 0, 0)
ref_h = np.exp(1j*phase)

class WrapNormal():
    def __init__(self, meas_prop):
        self.if_generator = fmcw_sys.modulation.get_modulation(meas_prop)
    def compute_likelihood(self, d, t, signal, linewidth):
        self.if_var = linewidth*2/meas_prop.sample_rate/(2*np.pi)
        tt, ifx = utils.estimate_if(t, signal, method='polar_discriminator', mirror=False)
        if_hat = self.if_generator.generate_freq_x(tt, np.expand_dims(d,1), 0, compute_jacobian=False, normalize_freq=True)
        if_hat = utils.wrap(if_hat, -1/2, 1/2)
        Z = np.zeros((len(tt)))
        K = 1
        extra_var = 0.
        for k in np.arange(-K, K):
            Z = Z + np.exp(-(ifx-if_hat+k)**2/(2*(self.if_var+extra_var)))
        nll = np.mean( -np.log(Z+1e-20), axis=1)
        return nll

        
tri_wn = WrapNormal(meas_prop)
param_name = 'sin_2e8_600_5'
n_cycle = 1

meas_prop = fmcw_sys.import_meas_prop_from_config(os.path.join('fmcw_sys', 'params.json'), param_name)
meas_prop.linewidth = L
dist_true = 300
vel_true = 0
tau_true = 2*dist_true/3e8
stft_window_size = 64

np.random.seed(276358)
meas_prop.assume_zero_velocity = False
sample_rate = meas_prop.get_sample_rate()
T = meas_prop.get_chirp_length()
t = np.arange(0, 2*T, 1./sample_rate)
fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
signal, second_output = fmcw_meas.generate(dist_true, vel_true, t, include_shot_noise=True)

phase = fmcw_meas.generate_phase(t, 0, 0)
ref2 = np.exp(1j*phase)

ref2_h_t = np.arange(0, 2*T, 1./sample_rate/6)
phase = fmcw_meas.generate_phase(ref2_h_t, 0, 0)
ref2_h = np.exp(1j*phase)

sin_wn = WrapNormal(meas_prop)

def gen_phase_noise_mix(ts, 
                    distance, 
                    velocity, 
                    linewidth: float,) -> np.ndarray[float]:

    if np.isscalar(distance) and np.isscalar(velocity):
        distance = np.array([distance])
        velocity = np.array([velocity])
    else:
        if len(distance) != len(velocity):
            raise ValueError("Distance and velocity must have same length")
        distance = np.array(distance)
        velocity = np.array(velocity)

    n_echo = len(distance)
    N_t = len(ts)

    tt = ts -  2*(np.reshape(distance,(-1,1))+np.reshape(velocity,(-1,1))*ts)/3e8

    ts_all = np.concatenate([ts, *[tt[i] for i in range(tt.shape[0])]])
    sort_idx = np.argsort(ts_all)
    ts_all = ts_all[sort_idx]

    phase_noise = np.zeros_like(ts_all)
    
    for i in range(1, 2*N_t):
        delay = ts_all[i] - ts_all[i-1]
        phase_noise[i] = phase_noise[i-1] + np.random.normal(0, scale=np.sqrt(np.abs(delay)*linewidth*2*np.pi))

    phase_noise_mix = np.zeros((n_echo, N_t))
    tx_idx = (sort_idx >= 0) * (sort_idx < N_t)
    phase_noise_tx = phase_noise[tx_idx]
    for n in range(n_echo):
        rx_idx = (sort_idx >= (n+1)*N_t) * (sort_idx < (n+2)*N_t)
        phase_noise_rx = phase_noise[rx_idx]
        phase_noise_mix[n] = phase_noise_tx - phase_noise_rx

    if n_echo == 1:
        phase_noise_mix = phase_noise_mix[0]

    return phase_noise_mix

distance_arr = np.linspace(300, 599, 10)
linewidth_arr = [1e5,]
n_sim = 100

tri_autocorr_mean = np.zeros((len(distance_arr), len(ref_h),))
sin_autocorr_mean = np.zeros((len(distance_arr), len(ref2_h),))
tri_autocorr_std = np.zeros((len(distance_arr), len(ref_h),))
sin_autocorr_std = np.zeros((len(distance_arr), len(ref2_h),))
pn_spectrum_mean = np.zeros((len(distance_arr), len(ref_h),))

lh_d = t/2*3e8
tri_lh_mean = np.zeros((len(linewidth_arr), len(lh_d),))
tri_lh_std = np.zeros((len(linewidth_arr), len(lh_d),))
sin_lh_mean = np.zeros((len(linewidth_arr), len(lh_d),))
sin_lh_std = np.zeros((len(linewidth_arr), len(lh_d),))
for id, l in enumerate(linewidth_arr):
    tri_autocorr_sim = np.zeros((n_sim, len(ref_h),))
    sin_autocorr_sim = np.zeros((n_sim, len(ref2_h),))
    pn_spectrum_sim = np.zeros((n_sim, len(ref2_h),))
    tri_lh_sim = np.zeros((n_sim, len(lh_d),))
    sin_lh_sim = np.zeros((n_sim, len(lh_d),))
    for i in range(n_sim):
        phase_noise = gen_phase_noise_mix(t, 300, 0, linewidth_arr[id])
        add_noise = ( np.random.randn(len(ref)) + 1j * np.random.randn(len(ref)) )  * 0.5
        pn_spectrum_sim[i] = np.abs(fft(np.exp(-1j*phase_noise), n=len(ref_h)))
        y1 = ref*np.exp(-1j*phase_noise) + add_noise
        y2 = ref2*np.exp(-1j*phase_noise) + add_noise
        ref_up_noisey = np.zeros_like(ref_h)
        ref2_up_noisey = np.zeros_like(ref2_h)
        ref_up_noisey[:6*len(ref):6] = y1
        ref2_up_noisey[:6*len(ref2):6] = y2
        
        
        tri_autocorr_sim[i] = np.abs(ifft(fft(ref_up_noisey) * np.conj(fft(ref_h))))
        sin_autocorr_sim[i] = np.abs(ifft(fft(ref2_up_noisey) * np.conj(fft(ref2_h))))
        tri_lh_sim[i] = tri_wn.compute_likelihood(lh_d, t, y1, 1e5)
        sin_lh_sim[i] = sin_wn.compute_likelihood(lh_d, t, y2, 1e5)
        
    tri_autocorr_mean[id] = np.mean(tri_autocorr_sim, axis=0)
    sin_autocorr_mean[id] = np.mean(sin_autocorr_sim, axis=0)
    tri_autocorr_std[id] = np.std(tri_autocorr_sim, axis=0)
    sin_autocorr_std[id] = np.std(sin_autocorr_sim, axis=0)
    pn_spectrum_mean[id] = np.mean(pn_spectrum_sim, axis=0)
    tri_lh_mean[id] = np.roll(np.mean(tri_lh_sim, axis=0), len(lh_d)//2)
    tri_lh_std[id] = np.roll(np.std(tri_lh_sim, axis=0), len(lh_d)//2)
    sin_lh_mean[id] = np.roll(np.mean(sin_lh_sim, axis=0), len(lh_d)//2)
    sin_lh_std[id] = np.roll(np.std(sin_lh_sim, axis=0), len(lh_d)//2)
    
idx = 0
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.errorbar((240-1200)/2400,
             np.roll(tri_autocorr_mean[idx], 2*1200)[2*240],
             np.roll(tri_autocorr_std[idx], 2*1200)[2*240],color="red",capsize=10, linewidth=4)
plt.errorbar((720-1200)/2400,
             np.roll(tri_autocorr_mean[idx], 2*1200)[2*720],
             np.roll(tri_autocorr_std[idx], 2*1200)[2*720],color="red",capsize=10, linewidth=4)
plt.errorbar((1200-1200)/2400,
             np.roll(tri_autocorr_mean[idx], 2*1200)[2*1200],
             np.roll(tri_autocorr_std[idx], 2*1200)[2*1200],color="red",capsize=10, linewidth=4)
plt.errorbar((1680-1200)/2400,
             np.roll(tri_autocorr_mean[idx], 2*1200)[2*1680],
             np.roll(tri_autocorr_std[idx], 2*1200)[2*1680],color="red",capsize=10, linewidth=4)
plt.errorbar((2160-1200)/2400,
             np.roll(tri_autocorr_mean[idx], 2*1200)[2*2160],
             np.roll(tri_autocorr_std[idx], 2*1200)[2*2160],color="red",capsize=10, linewidth=4, label="peak value deviation")
plt.plot((np.arange(len(tri_autocorr_mean[idx]))-2*1200)/2400/2, np.roll(tri_autocorr_mean[idx], 2*1200), color="black", linewidth=2.5, label="mean cross-correlation values")
plt.grid(linewidth=1)

plt.scatter((np.array([2*240, 2*720, 2*1200, 2*1680, 2*2160])-2*1200)/2400/2,
             np.roll(tri_autocorr_mean[idx], 2*1200)[[2*240, 2*720, 2*1200, 2*1680, 2*2160]], color="blue", s=50, zorder=10, label="mean peak values")

plt.yscale("log")
# plt.ylim(100, 1000)
plt.legend(fontsize=20)
plt.ylim(17, 4000)
plt.title("(i) Triangular Modulation", fontsize=25)
plt.xlabel("Difference from delay, $\\frac{t-\\tau}{2T}$", fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=17)

plt.subplot(122)


plt.errorbar((1200-1200)/2400,
             np.roll(sin_autocorr_mean[idx], 2*1200)[2*1200],
             np.roll(sin_autocorr_std[idx], 2*1200)[2*1200],color="red",capsize=10, linewidth=4)
plt.errorbar((978-2400)/4800,
             np.roll(sin_autocorr_mean[idx], 2*1200)[978],
             np.roll(sin_autocorr_std[idx], 2*1200)[978],color="red",capsize=10, linewidth=4)
plt.errorbar((1770-2400)/4800,
             np.roll(sin_autocorr_mean[idx], 2*1200)[1770],
             np.roll(sin_autocorr_std[idx], 2*1200)[1770],color="red",capsize=10, linewidth=4)
plt.errorbar((3030-2400)/4800,
             np.roll(sin_autocorr_mean[idx], 2*1200)[3030],
             np.roll(sin_autocorr_std[idx], 2*1200)[3030],color="red",capsize=10, linewidth=4)
plt.errorbar((3822-2400)/4800,
             np.roll(sin_autocorr_mean[idx], 2*1200)[3822],
             np.roll(sin_autocorr_std[idx], 2*1200)[3822],color="red",capsize=10, linewidth=4, label="peak value deviation")
plt.plot((np.arange(len(sin_autocorr_mean[idx]))-2*1200)/2400/2, np.roll(sin_autocorr_mean[idx], 2*1200), color="black", linewidth=2.5, label="mean cross-correlation values")
plt.grid(linewidth=1)
plt.scatter((np.array([978,1770,2400,3030,3822])-2400)/4800,
             np.roll(sin_autocorr_mean[idx], 2400)[[978,1770,2400,3030,3822]], color="blue", s=50, zorder=10, label="mean peak values")

plt.yscale("log")
plt.ylim(17, 4000)
plt.legend(fontsize=20)
plt.title("(ii) Sinusoidal Modulation", fontsize=25)
plt.xlabel("Difference from delay, $\\frac{t-\\tau}{2T}$", fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=17)
fig = plt.gcf()
fig.suptitle("(a) Deviation in Matched Filter Objective", fontsize=30)
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)

plt.figure(figsize=(16, 8))
plt.subplot(121)
print(len(tri_lh_mean[idx]))
for idx in range(len(linewidth_arr)-1, -1, -1):
    plt.plot((np.arange(len(tri_lh_mean[idx]))-400)/800, tri_lh_mean[idx],color="black")
    plt.errorbar((78-400)/800,
             tri_lh_mean[idx][78],
             tri_lh_std[idx][78],color="red",capsize=10, linewidth=4)
    plt.errorbar((240-400)/800,
                tri_lh_mean[idx][240],
                tri_lh_std[idx][240],color="red",capsize=10, linewidth=4)
    plt.errorbar((400-400)/800,
                tri_lh_mean[idx][400],
                tri_lh_std[idx][400],color="red",capsize=10, linewidth=4)
    plt.errorbar((560-400)/800,
                tri_lh_mean[idx][560],
                tri_lh_std[idx][560],color="red",capsize=10, linewidth=4)
    plt.errorbar((722-400)/800,
                tri_lh_mean[idx][722],
                tri_lh_std[idx][722],color="red",capsize=10, linewidth=4, label="local minimum deviation")
    plt.scatter((np.array([78,240,400,560,722])-400)/800,
             tri_lh_mean[idx][[78,240,400,560,722]], color="blue", s=50, zorder=10, label="mean local minimum values")

plt.grid(linewidth=1)
plt.ylim(18, 45)
plt.legend(fontsize=20)
plt.title("(i) Triangular Modulation", fontsize=25)
plt.xlabel("Difference from delay, $\\frac{t-\\tau}{2T}$", fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=17)

plt.subplot(122)
for idx in range(len(linewidth_arr)-1, -1, -1):
    plt.plot((np.arange(len(sin_lh_mean[idx]))-400)/800, sin_lh_mean[idx],color="black")
    plt.errorbar((154-400)/800,
             sin_lh_mean[idx][154],
             sin_lh_std[idx][154],color="red",capsize=10, linewidth=4)
    plt.errorbar((289-400)/800,
                sin_lh_mean[idx][289],
                sin_lh_std[idx][289],color="red",capsize=10, linewidth=4)
    plt.errorbar((400-400)/800,
                sin_lh_mean[idx][400],
                sin_lh_std[idx][400],color="red",capsize=10, linewidth=4)
    plt.errorbar((511-400)/800,
                sin_lh_mean[idx][511],
                sin_lh_std[idx][511],color="red",capsize=10, linewidth=4)
    plt.errorbar((646-400)/800,
                sin_lh_mean[idx][646],
                sin_lh_std[idx][646],color="red",capsize=10, linewidth=4, label="local minimum deviation")
    plt.scatter((np.array([154,289,400,511,646])-400)/800,
             sin_lh_mean[idx][[154,289,400,511,646]], color="blue", s=50, zorder=10, label="mean local minimum values")

plt.grid(linewidth=1)
plt.ylim(18, 45)
plt.legend(fontsize=20)
plt.title("(ii) Sinusoidal Modulation", fontsize=25)
plt.xlabel("Difference from delay, $\\frac{t-\\tau}{2T}$", fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=17)
fig = plt.gcf()
fig.suptitle("(b) Deviation in Wrapped Normal Objective", fontsize=30)
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)

plt.show()
