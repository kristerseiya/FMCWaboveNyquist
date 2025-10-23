
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))
import numpy as np
from scipy.fft import fft
from scipy.optimize import least_squares, curve_fit, bracket, minimize_scalar, minimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fmcw_sys import FMCWMeasurementProperties
import fmcw_sys
import time_freq
import optimizer
import utils

class Lorentzian_Regression_SquareError(optimizer.ObjectiveFunction):
    def __init__(self, t: np.ndarray, signal: np.ndarray, 
                 meas_prop: FMCWMeasurementProperties, shot_noise_var: float=0):
        self.meas_prop = meas_prop
        self.signal = signal
        self.periodogram = np.abs(fft(signal, norm='ortho'))**2
        self.periodogram /= np.amax(self.periodogram)
        N = len(self.signal)
        self.freq_arr = np.arange(0, sample_rate-sample_rate/N/2, sample_rate/N) / sample_rate
        self.shot_noise_var = shot_noise_var
        self.linewidth = meas_prop.get_linewidth() / sample_rate
        if (meas_prop.complex_available == False):
            periodogram = periodogram[:(N-N//2)]
            self.freq_arr = self.freq_arr[:(N-N//2)]

    def evaluate(self, x):
        N = len(self.signal)
        lorentz_func = x[:,[1]]*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x[:,[0]])**2 ) + self.shot_noise_var
        squared_error = np.sum( (self.periodogram - lorentz_func)**2, axis=-1)
        return squared_error
    
    def compute_gradient(self, x):
        x0 = x[:,[0]]
        x1 = x[:,[1]]
        N = len(self.signal)
        lorentz_func = x1*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0)**2 ) + self.shot_noise_var
        diff = (lorentz_func - self.periodogram)
        j0 = - x1*self.linewidth / ( np.pi ) * 2*(x0-self.freq_arr) / ( (self.freq_arr - x0)**2 + self.linewidth**2 )**2
        j1 = self.linewidth / ( (self.freq_arr - x0)**2 + self.linewidth**2 )
        x0_deriv = 2*np.sum(j0*diff, axis=-1)
        x1_deriv = 2*np.sum(j1*diff, axis=-1)
        return np.stack([x0_deriv, x1_deriv], axis=-1)


def lorentzian_regression(t: np.ndarray, signal: np.ndarray, linewidth: float|None=None, 
                       fit_linewidth: bool=False, shot_var: float=0, meas_prop: FMCWMeasurementProperties|None=None,
                       return_obj: bool=True):
    
    assert (np.ndim(t) == 1) and (np.ndim(signal) == 1)
    assert (linewidth is not None) or (meas_prop is not None)

    periodogram = np.abs(fft(signal, norm='ortho'))**2
    S = np.amax(periodogram)
    periodogram /= S
    shot_var /= S
    N = len(signal)
    freq_arr = np.arange(0, sample_rate-sample_rate/N/2, sample_rate/N) / sample_rate
    linewidth /= sample_rate
    x_init = np.array([[freq_arr[np.argmax(periodogram)], (np.amax(periodogram)-shot_var)*linewidth*np.pi],
                       [freq_arr[np.argmax(periodogram)]-0.5/N, (np.amax(periodogram)-shot_var)*linewidth*np.pi],
                       [freq_arr[np.argmax(periodogram)]+0.5/N, (np.amax(periodogram)-shot_var)*linewidth*np.pi]])
    
    cost_lorentz = Lorentzian_Regression_SquareError(t, signal, meas_prop, shot_var)
    
    def obj_func(x):
        return cost_lorentz.evaluate(np.expand_dims(x,axis=0))[0]
    def jac(x):
        return cost_lorentz.compute_gradient(np.expand_dims(x,axis=0))[0]
    x_hat = np.zeros((len(x_init),2))
    x_hat_cost = np.zeros((len(x_init),))
    for i, x0 in enumerate(x_init):
        res = minimize(obj_func, 
                    x0, jac=jac,
                    method="BFGS")
        x_hat[i,0] = res.x[0]
        x_hat[i,1] = res.x[1]
        x_hat_cost[i] = res.fun
                    
    # x_hat, x_hat_cost, gdtrack = optimizer.gradient_descent_backtrack_linesearch(x_init, cost_lorentz, max_n_iter=500, track=True, xtol=1e-7, c=1e-4, alpha0=1e-4)
    _, _, gdtrack = optimizer.gradient_descent_backtrack_linesearch(x_init, cost_lorentz, max_n_iter=500, track=True, xtol=1e-7, c=1e-4, alpha0=1e-4)
    x_hat_final = x_hat[np.argmin(x_hat_cost)]
    return x_hat_final, cost_lorentz, gdtrack

    

def lorentzian_fitting(t: np.ndarray, signal: np.ndarray, linewidth: float|None=None, 
                       fit_linewidth: bool=False, shot_var: float=0, meas_prop: FMCWMeasurementProperties|None=None,
                       return_obj: bool=True):
    
    assert (np.ndim(t) == 1) and (np.ndim(signal) == 1)
    assert (linewidth is not None) or (meas_prop is not None)

    if meas_prop is not None:
        linewidth = meas_prop.get_linewidth()
        
    N = len(signal)
    periodogram = np.abs(fft(signal, norm='ortho'))**2
    periodogram = periodogram.astype(float)
    periodogram /= np.amax(periodogram)
    if meas_prop is not None:
        sample_rate = meas_prop.get_sample_rate()
    else:
        sample_rate = 1/(t[1] - t[0])
    
    if (meas_prop is None) or (meas_prop.complex_available == False):
        periodogram = periodogram[:(N-N//2)]
        # f = np.fft.fftfreq(N, 1/sample_rate)
        f = np.arange(0, sample_rate-sample_rate/N/2, sample_rate/N)
        f = f[:(N-N//2)]
        # linewidth = 2*np.pi*linewidth #/ (2*np.pi)**2
    else:
        # f = np.fft.fftfreq(N, 1/sample_rate)
        f = np.arange(0, sample_rate-sample_rate/N/2, sample_rate/N)
    f = f / sample_rate
    linewidth = linewidth / sample_rate

    success = True

    if fit_linewidth:
        def lorentzian(f, x0, x1, x2):
            '''
            x0: beat frequency
            x1: amplitude^2
            x2: linewidth
            '''
            # return x1*x2 / ( (2*np.pi)**2 * (f - x0)**2 + x2**2 ) + awgn_var*N #+ x1*x2 / ( (2*np.pi)**2 * (fmax - x0)**2 + x2**2 )
            return x1*x2 / ( np.pi ) / ( x2**2 + (f-x0)**2 ) + shot_var
        #x0 = np.array([f[np.argmax(periodogram)], np.amax(periodogram-shot_var*N)*linewidth, linewidth])
        x_init = np.array([f[np.argmax(periodogram)], (np.amax(periodogram)-shot_var)*linewidth*np.pi, linewidth])
        x_lb = np.array([np.amin(f), 0, 0])
        x_ub = np.array([np.amax(f), np.inf, np.inf])

        def jacobian(f, x0, x1, x2):
            j0 = - x1*x2 / ( np.pi ) * 2*(x0-f) / ( (f - x0)**2 + x2**2 )**2
            j1 = x2 / ( (f - x0)**2 + x2**2 )
            j2 = (x1 * ( (f - x0)**2 + x2**2 ) - x1*x2 * 2*x2) / ( (f - x0)**2 + x2**2 )**2
            return np.stack([j0, j1, j2], axis=-1)

    else:
        def lorentzian(f, x0, x1):
            '''
            x0: beat frequency
            x1: amplitude
            '''
            # return x1*linewidth / ( (2*np.pi)**2 * (f - x0)**2 + linewidth**2 ) + shot_var*N #+ x1*linewidth / ( (2*np.pi)**2 * (fmax - x0)**2 + linewidth**2 )
            return x1*linewidth / ( np.pi ) / ( linewidth**2 + (f-x0)**2 ) + shot_var
        x_init = np.array([f[np.argmax(periodogram)], (np.amax(periodogram)-shot_var)*linewidth*np.pi ])
        print(x_init)
        x_lb = np.array([np.amin(f), 0])
        x_ub = np.array([np.amax(f), np.inf])

        def jacobian(f, x0, x1):
            j0 = - x1*linewidth / ( np.pi ) * 2*(x0-f) / ( (f - x0)**2 + linewidth**2 )**2 * 1e-9
            j1 = linewidth / ( (f - x0)**2 + linewidth**2 ) * 1e-9
            return np.stack([j0, j1], axis=-1)

    try:
        
        # def obj_func(x):
        #     return cost_lorentz.evaluate(np.expand_dims(x,axis=0), extra_var=0)[0]
        # def jac(x):
        #     return cost_lorentz.compute_gradient(np.expand_dims(x,axis=0), extra_var=0)[0]
        # res = minimize(obj_func, 
        #                 x_init, jac=jac,
        #                 method="BFGS")
        
        # def obj_func(x):
        #     return lorentzian(np.expand_dims(x,axis=0), extra_var=0)[0]
        # def jac(x):
        #     return cost_lorentz.compute_gradient(np.expand_dims(x,axis=0), extra_var=0)[0]
        # res = minimize(obj_func, 
        #                 x_init, jac=jac,
        #                 method="BFGS")
    
        popt, _ = curve_fit(lorentzian, f, periodogram, p0=x_init, bounds=(x_lb, x_ub), method='dogbox', jac=jacobian)
    except:
        success = False
        if fit_linewidth:
            popt = [-1,-1,-1]
        else:
            popt = [-1,-1]

    beat_freq_hat = popt[0]
    amp_scale_hat = popt[1]
    if fit_linewidth:
        linewidth_hat = popt[2]
    # if beat_freq_hat > sample_rate/2:
    #     beat_freq_hat = beat_freq_hat - sample_rate

    if fit_linewidth:
        return_value = [beat_freq_hat, amp_scale_hat, linewidth_hat]
    else:
        return_value = [beat_freq_hat, amp_scale_hat]

    if return_obj:
        return return_value, success, (lorentzian, f, periodogram, x_init)
    return return_value, success


# param_name = 'tri_2e8_1000_5'
param_name = 'tri_2e8_600_5'
# param_name = 'tri_2e8_600_5'
# param_name = 'sin_2e8_3000_5'
n_cycle = 1

meas_prop = fmcw_sys.import_meas_prop_from_config(utils.PARAMS_PATH, param_name)
meas_prop.assume_zero_velocity = True
meas_prop.complex_available = True
# meas_prop.linewidth = 1e5

dist_true = 100.57
vel_true = 0
tau_true = 2*dist_true/3e8
stft_window_size = 64

sample_rate = meas_prop.get_sample_rate()
T = meas_prop.get_chirp_length()
t = np.arange(0, 2*T*n_cycle, 1./sample_rate)
fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
signal, second_output = fmcw_meas.generate(dist_true, vel_true, t, 
                                                        #  phase_noise_seed=1346905138,
                                                        #  shot_noise_seed=4167281819,
                                                         include_shot_noise=True)

stft_plot = time_freq.short_time_fourier_transform(signal, stft_window_size, stft_window_size//4)
stft_freq = np.arange(0, 0.5+0.5/stft_window_size, 1./stft_window_size)

linewidth = meas_prop.get_linewidth()
shot_var = np.var(second_output)
cbf_region1, cbf_region2 = meas_prop.where_is_constant_beat_frequency(t, dist_true)

lorenz_param1, success1, obj1 = lorentzian_fitting(t[cbf_region1], signal[cbf_region1], 
                                                   linewidth=linewidth, shot_var=shot_var, 
                                                   meas_prop=meas_prop)
beat_freq1 = lorenz_param1[0] / sample_rate

lorenz_param2, obj2, gdtrack = lorentzian_regression(t[cbf_region1], signal[cbf_region1], 
                                    linewidth=linewidth, shot_var=shot_var, 
                                    meas_prop=meas_prop)
beat_freq2 = lorenz_param2[0] / sample_rate
print(lorenz_param2[0] * meas_prop.sample_rate / meas_prop.bandwidth * meas_prop.Tchirp * 3e8 / 2)

# lorenz_param2, success2, obj2 = lorentzian_fitting(t[cbf_region2], signal[cbf_region2], 
#                                                    linewidth=linewidth, shot_var=shot_var, 
#                                                    meas_prop=meas_prop)
# beat_freq2 = lorenz_param2[0] / sample_rate

# bandwidth = meas_prop.get_bandwidth()
# Tchirp = meas_prop.get_chirp_length()
# lambd_c = meas_prop.get_carrier_wavelength()
# gamma = meas_prop.get_bandwidth() / meas_prop.get_chirp_length()
# d_nyquist = sample_rate / 2 / (bandwidth / Tchirp) * 3e8 / 2
# v_nyquist = sample_rate / 2 * (lambd_c) / 2

# if meas_prop.is_complex_available():
#     # k = int(np.floor(distance/(d_nyquist*2)+1/2))
#     # freq1 = beat_freq1 + k
#     k1 = int(np.floor(dist_true/(d_nyquist*2)+0/(v_nyquist*2)+1/2))
#     k2 = int(np.floor(dist_true/(d_nyquist*2)-0/(v_nyquist*2)+1/2))
#     freq1 = beat_freq1 + k1
#     freq2 = -(beat_freq2 - k2)
# else:
#     # k = int(np.floor(distance/d_nyquist))
#     # freq1 = utils.unmirror(beat_freq1, 0, 0.5, n=k)
#     k1 = int(np.floor(dist_true/d_nyquist+0/v_nyquist))
#     k2 = int(np.floor(dist_true/d_nyquist-0/v_nyquist))
#     freq1 = utils.unmirror(beat_freq1, 0, 0.5, n=k1)
#     freq2 = utils.unmirror(beat_freq2, 0, 0.5, n=k2)
# d_hat = freq1 * sample_rate / gamma * 3e8 / 2
# v_hat = 0

f = obj1[0]
x_arr = obj1[1]
y = obj1[2]
x_init = obj1[3]
amp_max = np.amax([lorenz_param1[1], x_init[1], lorenz_param2[1]])
amp_min = np.amin([lorenz_param1[1], x_init[1], lorenz_param2[1]])
amp_med = 1/2*(amp_max+amp_min)
freq_max = np.amax([lorenz_param1[0], x_init[0], lorenz_param2[0]])
freq_min = np.amin([lorenz_param1[0], x_init[0], lorenz_param2[0]])
freq_med = 1/2*(freq_max+freq_min)
amp_arr = np.linspace(0, amp_max*2, 100)
freq_arr = np.linspace(x_init[0]-1/T/sample_rate, x_init[0]+1/T/sample_rate, 100)

freq_grid, amp_grid = np.meshgrid(freq_arr, amp_arr, indexing='ij')
lorentz_cost1 = np.zeros_like(freq_grid)
lorentz_cost2 = np.zeros_like(freq_grid)
for i in range(len(freq_arr)):
    for j in range(len(amp_arr)):
        lorentz_cost1[i,j] = np.sum( (y - f(x_arr, freq_grid[i,j], amp_grid[i,j]))**2 )
        lorentz_cost2[i,j] = obj2.evaluate(np.array([[freq_grid[i,j], amp_grid[i,j]]]))[0]

fig1 = plt.figure(1)
fig1.subplots_adjust(wspace=0.5)
ax11 = fig1.add_subplot(221)
ax12 = fig1.add_subplot(222)
ax13 = fig1.add_subplot(223)
ax14 = fig1.add_subplot(224)

contour1 = ax11.contourf(freq_grid, amp_grid, np.log(lorentz_cost1))
ax11.scatter(x_init[0], x_init[1], marker='x', color='green', s=200, linewidth=3)
ax11.scatter(lorenz_param1[0], lorenz_param1[1], marker='x', color='red', s=200, linewidth=3)
divider1 = make_axes_locatable(ax11)
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
fig1.colorbar(contour1, cax=cax1, orientation='vertical')
ax11.set_title('Scipy\'s NLLS')
ax11.set_xlabel('Normalized Beat Frequency (Hz)')
ax11.set_ylabel('Amplitude Scaling')

contour2 = ax12.contourf(freq_grid, amp_grid, np.log(lorentz_cost2))
ax12.scatter(gdtrack[0,0,0], gdtrack[0,1,0], marker='x', color='green', s=200, linewidth=3)
ax12.scatter(gdtrack[1,0,0], gdtrack[1,1,0], marker='x', color='green', s=200, linewidth=3)
ax12.scatter(gdtrack[2,0,0], gdtrack[2,1,0], marker='x', color='green', s=200, linewidth=3)
ax12.plot(gdtrack[0,0], gdtrack[0,1], color='magenta', linewidth=3)
ax12.plot(gdtrack[1,0], gdtrack[1,1], color='magenta', linewidth=3)
ax12.plot(gdtrack[2,0], gdtrack[2,1], color='magenta', linewidth=3)
ax12.scatter(lorenz_param2[0], lorenz_param2[1], marker='x', color='red', s=200, linewidth=3)
divider2 = make_axes_locatable(ax12)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
fig1.colorbar(contour2, cax=cax2, orientation='vertical')
ax12.set_title('Gradient Descent')
ax12.set_xlabel('Normalized Beat Frequency (Hz)')
ax12.set_ylabel('Amplitude Scaling')

ax13.plot(x_arr, np.log(y))
ax13.plot(x_arr, np.log(f(x_arr, lorenz_param1[0], lorenz_param1[1])))
ax13.set_xlabel('Normalized Frequency (Hz)')
ax13.set_ylabel('Normalized Amplitude')
ax13.set_title('Lorentzian Fit on Periodogram')

ax14.plot(x_arr, np.log(y), color='dodgerblue')
ax14.plot(x_arr, np.log(f(x_arr, lorenz_param2[0], lorenz_param2[1])), color='orange', linewidth=3)
ax14.set_xlabel('Normalized Frequency (Hz)')
ax14.set_ylabel('Normalized Amplitude')
ax13.set_title('Lorentzian Fit on Periodogram')

fig2 = plt.figure(2)
ax11 = fig2.add_subplot(111)
ax11.plot(np.arange(gdtrack.shape[-1]), gdtrack[0,-1])
ax11.plot(np.arange(gdtrack.shape[-1]), gdtrack[1,-1])
ax11.plot(np.arange(gdtrack.shape[-1]), gdtrack[2,-1])
plt.show()
