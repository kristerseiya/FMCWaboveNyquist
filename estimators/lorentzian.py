
import numpy as np
import copy
from scipy.fft import fft
from scipy.optimize import least_squares, curve_fit, bracket, minimize_scalar, minimize

from fmcw_sys import FMCWMeasurement, FMCWMeasurementProperties
import utils
import optimizer
from .estimators_base import Estimator, OracleEstimator

class Lorentzian_Regression_SquareError(optimizer.ObjectiveFunction):
    def __init__(self, signal: np.ndarray, 
                 meas_prop: FMCWMeasurementProperties, shot_noise_var: float=0):
        self.meas_prop = meas_prop
        self.sample_rate = meas_prop.get_sample_rate()
        self.signal = signal
        self.periodogram = np.abs(fft(signal, norm='ortho'))**2
        S = np.amax(self.periodogram)
        S = np.linalg.norm(signal)**2
        self.periodogram /= S
        N = len(self.signal)
        self.freq_arr = np.arange(0, 1-1/N/2, 1/N)
        self.shot_noise_var = shot_noise_var / S
        self.linewidth = meas_prop.get_linewidth() / self.sample_rate

    def evaluate(self, x):
        lorentz_func = x[:,[1]]*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x[:,[0]])**2 ) + \
                            x[:,[1]]*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x[:,[0]]+1)**2 ) + \
                            x[:,[1]]*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x[:,[0]]-1)**2 ) + self.shot_noise_var
        # lorentz_func = x[:,[1]]*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x[:,[0]])**2 ) + self.shot_noise_var
        squared_error = np.sum( (self.periodogram - lorentz_func)**2, axis=-1)
        return squared_error
    
    def compute_gradient(self, x):
        x0 = x[:,[0]]
        x1 = x[:,[1]]
        lorentz_func = x1*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0)**2 ) + \
                        x1*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0+1)**2 ) + \
                        x1*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0-1)**2 ) + self.shot_noise_var
        # lorentz_func = x1*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0)**2 ) + self.shot_noise_var
        diff = (lorentz_func - self.periodogram)
        j0 = - x1*self.linewidth / ( np.pi ) * 2*(x0-self.freq_arr) / ( (self.freq_arr - x0)**2 + self.linewidth**2 )**2 \
            - x1*self.linewidth / ( np.pi ) * 2*(x0-1-self.freq_arr) / ( (self.freq_arr - x0+1)**2 + self.linewidth**2 )**2 \
                - x1*self.linewidth / ( np.pi ) * 2*(x0+1-self.freq_arr) / ( (self.freq_arr - x0-1)**2 + self.linewidth**2 )**2
        # j0 = - x1*self.linewidth / ( np.pi ) * 2*(x0-self.freq_arr) / ( (self.freq_arr - x0)**2 + self.linewidth**2 )**2
        # j1 = self.linewidth / ( (self.freq_arr - x0)**2 + self.linewidth**2 )
        j1 = self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0)**2 ) + \
                        self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0+1)**2 ) + \
                        self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0-1)**2 )
        x0_deriv = 2*np.sum(j0*diff, axis=-1)
        x1_deriv = 2*np.sum(j1*diff, axis=-1)
        return np.stack([x0_deriv, x1_deriv], axis=-1)
    
class Lorentzian_Regression_SquareError2(optimizer.ObjectiveFunction):
    def __init__(self, signal: np.ndarray, 
                 meas_prop: FMCWMeasurementProperties, shot_noise_var: float=0):
        self.meas_prop = meas_prop
        self.sample_rate = meas_prop.get_sample_rate()
        self.signal = signal
        self.periodogram = np.abs(fft(signal, norm='ortho'))**2
        S = np.amax(self.periodogram)
        S = np.linalg.norm(signal)**2
        self.periodogram /= S
        N = len(self.signal)
        self.freq_arr = np.arange(0, 1-1/N/2, 1/N)
        self.shot_noise_var = shot_noise_var / S
        self.linewidth = meas_prop.get_linewidth() / self.sample_rate

    def evaluate(self, x):
        lorentz_func = np.exp(x[:,[1]])*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x[:,[0]])**2 ) + \
                            np.exp(x[:,[1]])*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x[:,[0]]+1)**2 ) + \
                            np.exp(x[:,[1]])*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x[:,[0]]-1)**2 ) + self.shot_noise_var
        # lorentz_func = x[:,[1]]*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x[:,[0]])**2 ) + self.shot_noise_var
        squared_error = np.sum( (self.periodogram - lorentz_func)**2, axis=-1)
        return squared_error
    
    def compute_gradient(self, x):
        x0 = x[:,[0]]
        x1 = x[:,[1]]
        lorentz_func = np.exp(x1)*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0)**2 ) + \
                        np.exp(x1)*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0+1)**2 ) + \
                        np.exp(x1)*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0-1)**2 ) + self.shot_noise_var
        # lorentz_func = x1*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0)**2 ) + self.shot_noise_var
        diff = (lorentz_func - self.periodogram)
        j0 = - np.exp(x1)*self.linewidth / ( np.pi ) * 2*(x0-self.freq_arr) / ( (self.freq_arr - x0)**2 + self.linewidth**2 )**2 \
            - np.exp(x1)*self.linewidth / ( np.pi ) * 2*(x0-1-self.freq_arr) / ( (self.freq_arr - x0+1)**2 + self.linewidth**2 )**2 \
                - np.exp(x1)*self.linewidth / ( np.pi ) * 2*(x0+1-self.freq_arr) / ( (self.freq_arr - x0-1)**2 + self.linewidth**2 )**2
        # j0 = - x1*self.linewidth / ( np.pi ) * 2*(x0-self.freq_arr) / ( (self.freq_arr - x0)**2 + self.linewidth**2 )**2
        j1 = np.exp(x1)*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0)**2 ) + \
                np.exp(x1)*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0+1)**2 ) + \
                np.exp(x1)*self.linewidth / ( np.pi ) / ( self.linewidth**2 + (self.freq_arr-x0-1)**2 )
        x0_deriv = 2*np.sum(j0*diff, axis=-1)
        x1_deriv = 2*np.sum(j1*diff, axis=-1)
        return np.stack([x0_deriv, x1_deriv], axis=-1)
    
def lorentzian_regression(signal: np.ndarray, meas_prop: FMCWMeasurementProperties, shot_var: float=0):
    
    assert (np.ndim(signal) == 1)
    assert (meas_prop is not None)

    periodogram = np.abs(fft(signal, norm='ortho'))**2
    N = len(signal)
    sample_rate = meas_prop.get_sample_rate()
    linewidth = meas_prop.get_linewidth()
    freq_arr = np.arange(0, sample_rate-sample_rate/N/2, sample_rate/N) / sample_rate
    linewidth /= sample_rate
    x_init = np.array([[freq_arr[np.argmax(periodogram)], (np.amax(periodogram)-shot_var)*linewidth*np.pi],
                       [freq_arr[np.argmax(periodogram)]-0.5/N, (np.amax(periodogram)-shot_var)*linewidth*np.pi],
                       [freq_arr[np.argmax(periodogram)]+0.5/N, (np.amax(periodogram)-shot_var)*linewidth*np.pi]])
    
    cost_lorentz = Lorentzian_Regression_SquareError(signal, meas_prop, shot_var)
                
    
    # x_hat, x_hat_cost = optimizer.gradient_descent_backtrack_linesearch(x_init, cost_lorentz, max_n_iter=500, track=False, xtol=1e-7, c=1e-4, alpha0=1e-4, beta=0.1)
    x_hat, x_hat_cost = optimizer.gradient_descent_backtrack_linesearch(x_init, cost_lorentz, max_n_iter=500, track=False, xtol=1e-7, c=1e-4, alpha0=1e-4, beta=0.1)
    x_hat_final = x_hat[np.argmin(x_hat_cost)]
    x_hat_final[0] = (np.mod(x_hat_final[0]+0.5,1)-0.5) * sample_rate
    return x_hat_final

def lorentzian_regression_faster(signal: np.ndarray, meas_prop: FMCWMeasurementProperties, shot_var: float=0):
    
    assert (np.ndim(signal) == 1)
    assert (meas_prop is not None)

    periodogram = np.abs(fft(signal, norm='ortho'))**2
    N = len(signal)
    sample_rate = meas_prop.get_sample_rate()
    linewidth = meas_prop.get_linewidth()
    freq_arr = np.arange(0, sample_rate-sample_rate/N/2, sample_rate/N) / sample_rate
    linewidth /= sample_rate
    x_init = np.array([[freq_arr[np.argmax(periodogram)], (np.amax(periodogram)-shot_var)*linewidth*np.pi],
                       [freq_arr[np.argmax(periodogram)]-0.5/N, (np.amax(periodogram)-shot_var)*linewidth*np.pi],
                       [freq_arr[np.argmax(periodogram)]+0.5/N, (np.amax(periodogram)-shot_var)*linewidth*np.pi]])
    # x_init = np.array([[freq_arr[np.argmax(periodogram)], np.log((np.amax(periodogram)-shot_var)*linewidth*np.pi)],
    #                    [freq_arr[np.argmax(periodogram)]-0.5/N, np.log((np.amax(periodogram)-shot_var)*linewidth*np.pi)],
    #                    [freq_arr[np.argmax(periodogram)]+0.5/N, np.log((np.amax(periodogram)-shot_var)*linewidth*np.pi)]])
    
    cost_lorentz = Lorentzian_Regression_SquareError(signal, meas_prop, shot_var)
                
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
        
    x_hat_final = x_hat[np.argmin(x_hat_cost)]
    x_hat_final[0] = (np.mod(x_hat_final[0]+0.5,1)-0.5) * sample_rate
    return x_hat_final

def lorentzian_fitting(t: np.ndarray, signal: np.ndarray, linewidth: float|None=None, 
                       fit_linewidth: bool=False, shot_var: float=0, meas_prop: FMCWMeasurementProperties|None=None):
    
    assert (np.ndim(t) == 1) and (np.ndim(signal) == 1)
    assert (linewidth is not None) or (meas_prop is not None)

    if meas_prop is not None:
        linewidth = meas_prop.get_linewidth()
        
    N = len(signal)
    periodogram = np.abs(fft(signal, norm='ortho'))**2
    periodogram = periodogram.astype(float)
    if meas_prop is not None:
        sample_rate = meas_prop.get_sample_rate()
    else:
        sample_rate = 1/(t[1] - t[0])
        
    linewidth /= sample_rate
    
    if (meas_prop is None) or (meas_prop.complex_available == False):
        periodogram = periodogram[:(N-N//2)]
        # f = np.fft.fftfreq(N, 1/sample_rate)
        f = np.arange(0, sample_rate-sample_rate/N/2, sample_rate/N) / sample_rate
        f = f[:(N-N//2)]
        # linewidth = 2*np.pi*linewidth #/ (2*np.pi)**2
    else:
        # f = np.fft.fftfreq(N, 1/sample_rate)
        f = np.arange(0, sample_rate-sample_rate/N/2, sample_rate/N) / sample_rate

    success = True

    if fit_linewidth:
        def lorentzian(f, x0, x1, x2):
            '''
            x0: beat frequency
            x1: amplitude^2
            x2: linewidth
            '''
            # return x1*x2 / ( (2*np.pi)**2 * (f - x0)**2 + x2**2 ) + awgn_var*N #+ x1*x2 / ( (2*np.pi)**2 * (fmax - x0)**2 + x2**2 )
            return x1*x2 / ( np.pi ) / ( x2**2 + (f-x0)**2 ) * N**2 + shot_var
        #x0 = np.array([f[np.argmax(periodogram)], np.amax(periodogram-shot_var*N)*linewidth, linewidth])
        x_init = np.array([f[np.argmax(periodogram)], (np.amax(periodogram)-shot_var)*linewidth/(N**2)*np.pi, linewidth])

        x_lb = np.array([np.amin(f), 0, 0])
        x_ub = np.array([np.amax(f), np.inf, np.inf])

        def jacobian(f, x0, x1, x2):
            j0 = - x1*x2 / ( np.pi ) * 2*(x0-f) / ( (f - x0)**2 + x2**2 )**2 * N**2
            j1 = x2 / ( (f - x0)**2 + x2**2 ) * N**2
            j2 = (x1 * ( (f - x0)**2 + x2**2 ) - x1*x2 * 2*x2) / ( (f - x0)**2 + x2**2 )**2
            return np.stack([j0, j1, j2], axis=-1)

    else:
        def lorentzian(f, x0, x1):
            '''
            x0: beat frequency
            x1: amplitude
            '''
            # return x1*linewidth / ( (2*np.pi)**2 * (f - x0)**2 + linewidth**2 ) + shot_var*N #+ x1*linewidth / ( (2*np.pi)**2 * (fmax - x0)**2 + linewidth**2 )
            return x1*linewidth / ( np.pi ) / ( linewidth**2 + (f-x0)**2 ) * N**2 + shot_var
        x_init = np.array([f[np.argmax(periodogram)], (np.amax(periodogram)-shot_var)*linewidth/(N**2)*np.pi])

        x_lb = np.array([np.amin(f), 0])
        x_ub = np.array([np.amax(f), np.inf])

        def jacobian(f, x0, x1):
            j0 = - x1*linewidth / ( np.pi ) * 2*(x0-f) / ( (f - x0)**2 + linewidth**2 )**2 * N**2
            j1 = linewidth / ( (f - x0)**2 + linewidth**2 ) * N**2
            return np.stack([j0, j1], axis=-1)

    try:
        popt, _ = curve_fit(lorentzian, f, periodogram, p0=x_init, bounds=(x_lb, x_ub), method='trf', jac=jacobian)
    except:
        success = False
        if fit_linewidth:
            popt = [-1,-1,-1]
        else:
            popt = [-1,-1]

    beat_freq_hat = popt[0] * sample_rate
    amp_scale_hat = popt[1]
    if fit_linewidth:
        linewidth_hat = popt[2]
    if beat_freq_hat > sample_rate/2:
        beat_freq_hat = beat_freq_hat - sample_rate

    if fit_linewidth:
        return_value = (beat_freq_hat, amp_scale_hat, linewidth_hat)
    else:
        return_value = (beat_freq_hat, amp_scale_hat)

    return return_value, success

def max_periodogram(t: np.ndarray, signal: np.ndarray, method='fine', meas_prop: FMCWMeasurementProperties|None=None):

    N = len(signal)
    periodogram = np.abs(fft(signal, norm='ortho'))**2
    periodogram = periodogram.astype(float)
    if meas_prop is not None:
        sample_rate = meas_prop.get_sample_rate()
    else:
        sample_rate = 1/(t[1] - t[0])

    if (meas_prop is None) or (meas_prop.complex_available == False):
        periodogram = periodogram[:(N-N//2)]
        f = np.fft.fftfreq(N, 1/sample_rate)
        f = f[:(N-N//2)]
        # linewidth = 2*np.pi*linewidth #/ (2*np.pi)**2
    else:
        f = np.fft.fftfreq(N, 1/sample_rate)

    max_idx = np.argmax(periodogram)
    if method == 'coarse':
        pdgm_max = f[max_idx]
        success = True
    elif method == 'fine':
        if max_idx == 0:
            brckt = (f[0], f[1])
        elif max_idx == (N-N//2-1):
            brckt = (f[max_idx-1], f[max_idx])
        else:
            # scipy bracket
            xa, xb, xc, _, _, _, _ = bracket(lambda f: -np.abs(np.exp(1j * 2*np.pi * f * t) @ signal)**2, xa=f[max_idx-1], xb=f[max_idx])
            brckt = (xa, xb, xc)
            # bracket = (freqs[np.argmax(pdgm)-1], freqs[np.argmax(pdgm)], freqs[np.argmax(pdgm)+1])

        try:
            opt_res = minimize_scalar(lambda f: -np.abs(np.exp(1j * 2*np.pi * f * t) @ signal)**2, bracket=brckt, method='brent')
            pdgm_max = opt_res.x
            success = True
        except:
            pdgm_max = f[max_idx]
            success = False
    return pdgm_max, success

class LorentzianRegressor(Estimator):
    def __init__(self, meas_prop: FMCWMeasurementProperties, 
                 ignore_quadrature=False, remap=True, method="gd"):

        if not (meas_prop.get_modulation_type() == 'triangle'):
            raise ValueError('modulation type not compatible for this estimator')
        
        self.meas_prop = copy.deepcopy(meas_prop)
        self.ignore_quadrature = ignore_quadrature
        if self.ignore_quadrature:
            self.meas_prop.complex_available = False
        self.sample_rate = self.meas_prop.get_sample_rate()
        self.bandwidth = self.meas_prop.get_bandwidth()
        self.Tchirp = self.meas_prop.get_chirp_length()
        self.lambd_c = self.meas_prop.get_carrier_wavelength()
        self.linewidth = self.meas_prop.get_linewidth()
        self.assume_zero_velocity = self.meas_prop.is_zero_velocity()
        self.complex_available = self.meas_prop.is_complex_available()
        self.test_flag = False
        self.remap = remap
        self.method = method

    def estimate(self, t: np.ndarray, mixed_signal: np.ndarray, second_output: np.ndarray|None=None):

        assert (np.ndim(t)==1) and (np.ndim(mixed_signal)==1)

        if self.assume_zero_velocity and ( (t[-1] + 2/self.sample_rate - t[0])<(self.Tchirp) ):
            raise RuntimeError('not enough observations')
        elif not self.assume_zero_velocity and ( (t[-1] + 2/self.sample_rate - t[0])<(2*self.Tchirp) ):
            raise RuntimeError('not enough observations')

        if self.ignore_quadrature:
            mixed_signal = np.real(mixed_signal)
            if second_output is not None:
                second_output = np.real(second_output)

        if second_output is None:
            shot_var = 0
        else:
            shot_var = np.var(second_output)

        d_nyquist = self.sample_rate / 2 / (self.bandwidth / self.Tchirp) * 3e8 / 2
        v_nyquist = self.sample_rate / 2 * (self.lambd_c) / 2

        # meas_start = (tau + 10/self.sample_rate)
        # meas_end = self.Tchirp
        # meas1_idx = (t > meas_start) * (t < meas_end)
        # meas2_idx = (t > (self.Tchirp + meas_start)) * (t < (self.Tchirp + meas_end))

        meas1_idx, meas2_idx = self.meas_prop.where_is_constant_beat_frequency(t, 120)
        meas1_idx = (t>=0) * (t<self.Tchirp)
        meas2_idx = (t>=self.Tchirp) * (t<(2*self.Tchirp))
        if ( np.sum(meas1_idx) < 10 ) or ( np.sum(meas2_idx) < 10):
            success1 = False
            success2 = False
        else:
            if self.method == "gd":
                lorenz_param1 = lorentzian_regression(mixed_signal[meas1_idx], self.meas_prop, shot_var=shot_var)
                success1 = True
                beat_freq1 = lorenz_param1[0] / self.sample_rate
                lorenz_param2 = lorentzian_regression(mixed_signal[meas2_idx], self.meas_prop, shot_var=shot_var)
                success2 = True
                beat_freq2 = lorenz_param2[0] / self.sample_rate
            elif self.method == "LBFGS":
                lorenz_param1 = lorentzian_regression_faster(mixed_signal[meas1_idx], self.meas_prop, shot_var=shot_var)
                success1 = True
                beat_freq1 = lorenz_param1[0] / self.sample_rate
                lorenz_param2 = lorentzian_regression_faster(mixed_signal[meas2_idx], self.meas_prop, shot_var=shot_var)
                success2 = True
                beat_freq2 = lorenz_param2[0] / self.sample_rate
            # lorenz_param1 = lorentzian_fitting(t, mixed_signal[meas1_idx], self.meas_prop.linewidth, 
            #            fit_linewidth=False, shot_var=shot_var, meas_prop=self.meas_prop)[0]
            # success1 = True
            # beat_freq1 = lorenz_param1[0] / self.sample_rate
            # lorenz_param2 = lorentzian_fitting(t, mixed_signal[meas2_idx], self.meas_prop.linewidth, 
            #            fit_linewidth=False, shot_var=shot_var, meas_prop=self.meas_prop)[0]
            # success2 = True
            # beat_freq2 = lorenz_param2[0] / self.sample_rate
            
        if self.assume_zero_velocity and success1 and success2:
            gamma = self.bandwidth / self.Tchirp
            if self.complex_available:
                freq1 = beat_freq1
                freq2 = beat_freq2
                if self.remap:
                    if  ( (freq1 + freq2) > (0.5) ) * ( (freq1 - freq2) >= 0 ):
                        freq1 -= 1
                    elif ( (freq1 + freq2) > (0.5) ) * ( (freq1 - freq2) < 0 ):
                        freq2 -= 1
                    elif ( (freq1 + freq2) < -(0.5) ) * ( (freq1 - freq2) >= 0 ):
                        freq2 += 1
                    elif ( (freq1 + freq2) < -(0.5) ) * ( (freq1 - freq2) < 0 ):
                        freq1 += 1
                    if (freq1 - freq2) < 0:
                        freq1 += 1
                        freq2 -= 1
                    # case2 = ( (freq1 + freq2) > (0.5) ) * ( (freq1 - freq2) >= 0 )
                    # case3 = ( (freq1 + freq2) > (0.5) ) * ( (freq1 - freq2) < 0 )
                    # case4 = ( (freq1 + freq2) < -(0.5) ) * ( (freq1 - freq2) >= 0 )
                    # case5 = ( (freq1 + freq2) < -(0.5) ) * ( (freq1 - freq2) < 0 )
                    # freq1[case2] -= 1
                    # freq2[case3] -= 1
                    # freq2[case4] += 1
                    # freq1[case5] += 1
                    # case6 = ( (freq1 - freq2) < 0 )
                    # freq1[case6] += 1
                    # freq2[case6] -= 1
            else:
                freq1 = beat_freq1
                freq2 = -beat_freq2
            d_hat = (freq1-freq2)/2 * self.sample_rate / gamma * 3e8 / 2
            v_hat = 0
        elif not self.assume_zero_velocity and success1 and success2:
            gamma = self.bandwidth / self.Tchirp
            if self.complex_available:
                freq1 = beat_freq1
                freq2 = beat_freq2
                if self.remap:
                    if  ( (freq1 + freq2) > (0.5) ) * ( (freq1 - freq2) >= 0 ):
                        freq1 -= 1
                    elif ( (freq1 + freq2) > (0.5) ) * ( (freq1 - freq2) < 0 ):
                        freq2 -= 1
                    elif ( (freq1 + freq2) < -(0.5) ) * ( (freq1 - freq2) >= 0 ):
                        freq2 += 1
                    elif ( (freq1 + freq2) < -(0.5) ) * ( (freq1 - freq2) < 0 ):
                        freq1 += 1
                    
                    if (freq1 - freq2) < 0:
                        freq1 += 1
                        freq2 -= 1
                    # case2 = ( (freq1 + freq2) > (0.5) ) * ( (freq1 - freq2) >= 0 )
                    # case3 = ( (freq1 + freq2) > (0.5) ) * ( (freq1 - freq2) < 0 )
                    # case4 = ( (freq1 + freq2) < -(0.5) ) * ( (freq1 - freq2) >= 0 )
                    # case5 = ( (freq1 + freq2) < -(0.5) ) * ( (freq1 - freq2) < 0 )
                    # freq1[case2] -= 1
                    # freq2[case3] -= 1
                    # freq2[case4] += 1
                    # freq1[case5] += 1
                    # case6 = ( (freq1 - freq2) < 0 )
                    # freq1[case6] += 1
                    # freq2[case6] -= 1
            else:
                freq1 = beat_freq1
                freq2 = -beat_freq2
            d_hat = (freq1-freq2) / 2 * self.sample_rate / gamma * 3e8 / 2
            v_hat = (freq1+freq2) / 2 * self.sample_rate * self.lambd_c / 2
        else:
            d_hat = -111
            v_hat = -111

        x_hat = np.array([d_hat, v_hat])

        return x_hat
    
    def set_test_flag(self, test: bool):
        self.test_flag = test

class OracleLorentzianRegressor(OracleEstimator):
    def __init__(self, meas_prop: FMCWMeasurementProperties, ignore_quadrature=False, method='lorentzian'):

        if not (meas_prop.get_modulation_type() == 'triangle'):
            raise ValueError('modulation type not compatible for this estimator')
        
        self.meas_prop = copy.deepcopy(meas_prop)
        self.ignore_quadrature = ignore_quadrature
        if self.ignore_quadrature:
            self.meas_prop.complex_available = False
        self.sample_rate = self.meas_prop.get_sample_rate()
        self.bandwidth = self.meas_prop.get_bandwidth()
        self.Tchirp = self.meas_prop.get_chirp_length()
        self.lambd_c = self.meas_prop.get_carrier_wavelength()
        self.linewidth = self.meas_prop.get_linewidth()
        self.assume_zero_velocity = self.meas_prop.is_zero_velocity()
        self.complex_available = self.meas_prop.is_complex_available()
        self.method = method
        self.test_flag = False

    def estimate(self, x: np.ndarray, t: np.ndarray, mixed_signal: np.ndarray, second_output: np.ndarray|None=None):

        assert (np.ndim(x)==1) and (np.ndim(t)==1) and (np.ndim(mixed_signal)==1)

        if self.assume_zero_velocity and ( (t[-1] + 2/self.sample_rate - t[0])<(self.Tchirp) ):
            raise RuntimeError('not enough observations')
        elif not self.assume_zero_velocity and ( (t[-1] + 2/self.sample_rate - t[0])<(2*self.Tchirp) ):
            raise RuntimeError('not enough observations')

        if self.ignore_quadrature:
            mixed_signal = np.real(mixed_signal)
            if second_output is not None:
                second_output = np.real(second_output)

        if second_output is None:
            shot_var = 0
        else:
            shot_var = np.var(second_output)

        distance = x[0]
        veloctiy = x[1]
        d_nyquist = self.sample_rate / 2 / (self.bandwidth / self.Tchirp) * 3e8 / 2
        v_nyquist = self.sample_rate / 2 * (self.lambd_c) / 2
        tau = distance * 2 / 3e8 

        # meas_start = (tau + 10/self.sample_rate)
        # meas_end = self.Tchirp
        # meas1_idx = (t > meas_start) * (t < meas_end)
        # meas2_idx = (t > (self.Tchirp + meas_start)) * (t < (self.Tchirp + meas_end))

        cbf_region1, cbf_region2 = self.meas_prop.where_is_constant_beat_frequency(t, distance)
        meas1_idx = cbf_region1
        meas2_idx = cbf_region2

        if ( np.sum(meas1_idx) < 10 ) or ( np.sum(meas2_idx) < 10):
            success1 = False
            success2 = False
        else:
            lorenz_param1 = lorentzian_regression(mixed_signal[meas1_idx], self.meas_prop, shot_var=shot_var)
            success1 = True
            beat_freq1 = lorenz_param1[0] / self.sample_rate
            lorenz_param2 = lorentzian_regression(mixed_signal[meas2_idx], self.meas_prop, shot_var=shot_var)
            success2 = True
            beat_freq2 = lorenz_param2[0] / self.sample_rate

        if self.assume_zero_velocity and success1 and success2:
            gamma = self.bandwidth / self.Tchirp
            if self.complex_available:
                # k = int(np.floor(distance/(d_nyquist*2)+1/2))
                # freq1 = beat_freq1 + k
                k1 = int(np.floor(distance/(d_nyquist*2)+0/(v_nyquist*2)+1/2))
                k2 = int(np.floor(distance/(d_nyquist*2)-0/(v_nyquist*2)+1/2))
                freq1 = beat_freq1 + k1
                freq2 = -(beat_freq2 - k2)
            else:
                # k = int(np.floor(distance/d_nyquist))
                # freq1 = utils.unmirror(beat_freq1, 0, 0.5, n=k)
                k1 = int(np.floor(distance/d_nyquist+0/v_nyquist))
                k2 = int(np.floor(distance/d_nyquist-0/v_nyquist))
                freq1 = utils.unmirror(beat_freq1, 0, 0.5, n=k1)
                freq2 = utils.unmirror(beat_freq2, 0, 0.5, n=k2)
            d_hat = freq1 * self.sample_rate / gamma * 3e8 / 2
            v_hat = 0
        elif not self.assume_zero_velocity and success1 and success2:
            gamma = self.bandwidth / self.Tchirp
            if self.complex_available:
                k1 = int(np.floor(distance/(d_nyquist*2)+veloctiy/(v_nyquist*2)+1/2))
                k2 = int(np.floor(distance/(d_nyquist*2)-veloctiy/(v_nyquist*2)+1/2))
                freq1 = beat_freq1 + k1
                freq2 = -(beat_freq2 - k2)
            else:
                k1 = int(np.floor(distance/d_nyquist+veloctiy/v_nyquist))
                k2 = int(np.floor(distance/d_nyquist-veloctiy/v_nyquist))
                freq1 = utils.unmirror(beat_freq1, 0, 0.5, n=k1)
                freq2 = utils.unmirror(beat_freq2, 0, 0.5, n=k2)
            d_hat = (freq1+freq2) / 2 * self.sample_rate / gamma * 3e8 / 2
            v_hat = (freq1-freq2) / 2 * self.sample_rate * self.lambd_c / 2
        else:
            d_hat = -111
            v_hat = -111

        x_hat = np.array([d_hat, v_hat])

        return x_hat
    
    def set_test_flag(self, test: bool):
        self.test_flag = test

class ConstantFrequencyEstimator(OracleEstimator):
    def __init__(self, meas_prop: FMCWMeasurementProperties, ignore_quadrature=False, method='lorentzian'):

        if not (meas_prop.get_modulation_type() == 'triangle'):
            raise ValueError('modulation type not compatible for this estimator')
        
        self.meas_prop = copy.deepcopy(meas_prop)
        self.ignore_quadrature = ignore_quadrature
        if self.ignore_quadrature:
            self.meas_prop.complex_available = False
        self.sample_rate = self.meas_prop.get_sample_rate()
        self.bandwidth = self.meas_prop.get_bandwidth()
        self.Tchirp = self.meas_prop.get_chirp_length()
        self.lambd_c = self.meas_prop.get_carrier_wavelength()
        self.linewidth = self.meas_prop.get_linewidth()
        self.assume_zero_velocity = self.meas_prop.is_zero_velocity()
        self.complex_available = self.meas_prop.is_complex_available()
        self.method = method
        self.test_flag = False

    def estimate(self, x: np.ndarray, t: np.ndarray, mixed_signal: np.ndarray, second_output: np.ndarray|None=None):

        assert (np.ndim(x)==1) and (np.ndim(t)==1) and (np.ndim(mixed_signal)==1)

        if self.assume_zero_velocity and ( (t[-1] + 2/self.sample_rate - t[0])<(self.Tchirp) ):
            raise RuntimeError('not enough observations')
        elif not self.assume_zero_velocity and ( (t[-1] + 2/self.sample_rate - t[0])<(2*self.Tchirp) ):
            raise RuntimeError('not enough observations')

        if self.ignore_quadrature:
            mixed_signal = np.real(mixed_signal)
            if second_output is not None:
                second_output = np.real(second_output)

        if second_output is None:
            shot_var = 0
        else:
            shot_var = np.var(second_output)

        distance = x[0]
        veloctiy = x[1]
        d_nyquist = self.sample_rate / 2 / (self.bandwidth / self.Tchirp) * 3e8 / 2
        v_nyquist = self.sample_rate / 2 * (self.lambd_c) / 2
        tau = distance * 2 / 3e8 

        # meas_start = (tau + 10/self.sample_rate)
        # meas_end = self.Tchirp
        # meas1_idx = (t > meas_start) * (t < meas_end)
        # meas2_idx = (t > (self.Tchirp + meas_start)) * (t < (self.Tchirp + meas_end))

        cbf_region1, cbf_region2 = self.meas_prop.where_is_constant_beat_frequency(t, distance)
        meas1_idx = cbf_region1
        meas2_idx = cbf_region2

        if ( np.sum(meas1_idx) < 10 ) or ( np.sum(meas2_idx) < 10):
            success1 = False
            success2 = False
        else:
            if self.method == 'lorentzian':
                lorenz_param1 = lorentzian_regression(mixed_signal[meas1_idx], self.meas_prop, shot_var=shot_var)
                success1 = True
                beat_freq1 = lorenz_param1[0] / self.sample_rate
                lorenz_param2 = lorentzian_regression(mixed_signal[meas2_idx], self.meas_prop, shot_var=shot_var)
                success2 = True
                beat_freq2 = lorenz_param2[0] / self.sample_rate
            elif self.method == 'maxpd':
                pdgm_max1, success1 = max_periodogram(t[meas1_idx], mixed_signal[meas1_idx], 
                                                                method='fine', meas_prop=self.meas_prop)
                beat_freq1 = pdgm_max1 / self.sample_rate
                pdgm_max2, success2 = max_periodogram(t[meas2_idx], mixed_signal[meas2_idx], 
                                                                method='fine', meas_prop=self.meas_prop)
                beat_freq2 = pdgm_max2 / self.sample_rate

        if self.assume_zero_velocity and success1 and success2:
            gamma = self.bandwidth / self.Tchirp
            if self.complex_available:
                # k = int(np.floor(distance/(d_nyquist*2)+1/2))
                # freq1 = beat_freq1 + k
                k1 = int(np.floor(distance/(d_nyquist*2)+0/(v_nyquist*2)+1/2))
                k2 = int(np.floor(distance/(d_nyquist*2)-0/(v_nyquist*2)+1/2))
                freq1 = beat_freq1 + k1
                freq2 = -(beat_freq2 - k2)
            else:
                # k = int(np.floor(distance/d_nyquist))
                # freq1 = utils.unmirror(beat_freq1, 0, 0.5, n=k)
                k1 = int(np.floor(distance/d_nyquist+0/v_nyquist))
                k2 = int(np.floor(distance/d_nyquist-0/v_nyquist))
                freq1 = utils.unmirror(beat_freq1, 0, 0.5, n=k1)
                freq2 = utils.unmirror(beat_freq2, 0, 0.5, n=k2)
            d_hat = freq1 * self.sample_rate / gamma * 3e8 / 2
            v_hat = 0
        elif not self.assume_zero_velocity and success1 and success2:
            gamma = self.bandwidth / self.Tchirp
            if self.complex_available:
                k1 = int(np.floor(distance/(d_nyquist*2)+veloctiy/(v_nyquist*2)+1/2))
                k2 = int(np.floor(distance/(d_nyquist*2)-veloctiy/(v_nyquist*2)+1/2))
                freq1 = beat_freq1 + k1
                freq2 = -(beat_freq2 - k2)
            else:
                k1 = int(np.floor(distance/d_nyquist+veloctiy/v_nyquist))
                k2 = int(np.floor(distance/d_nyquist-veloctiy/v_nyquist))
                freq1 = utils.unmirror(beat_freq1, 0, 0.5, n=k1)
                freq2 = utils.unmirror(beat_freq2, 0, 0.5, n=k2)
            d_hat = (freq1+freq2) / 2 * self.sample_rate / gamma * 3e8 / 2
            v_hat = (freq1-freq2) / 2 * self.sample_rate * self.lambd_c / 2
        else:
            d_hat = -111
            v_hat = -111

        x_hat = np.array([d_hat, v_hat])

        return x_hat
    
    def set_test_flag(self, test: bool):
        self.test_flag = test
    