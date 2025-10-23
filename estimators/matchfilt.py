
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
import copy

import fmcw_sys
from fmcw_sys import FMCWMeasurementProperties
from .estimators_base import Estimator
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import circulant

def matched_filter(meas_prop: FMCWMeasurementProperties, 
                   mixed_signal: np.ndarray,
                   optimize: bool=False):
    fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
    sample_rate = meas_prop.get_sample_rate()
    bandwidth = meas_prop.get_bandwidth()
    if bandwidth > sample_rate/2:
        K = int(np.ceil(bandwidth/sample_rate*2))
    else:
        K = 1
    T = meas_prop.get_chirp_length()
    n_cycle = int(np.ceil( len(mixed_signal) / (2*T*sample_rate) ))
    ta = np.arange(0, 2*T*n_cycle, 1/sample_rate/K)
    xa = np.zeros((len(ta)), dtype=mixed_signal.dtype)
    N = len(mixed_signal)
    N = min(len(mixed_signal), len(xa[0:N*K:K]))
    xa[0:N*K:K] = mixed_signal[:N]
    phase_tx = fmcw_meas.generate_phase(ta, 0, 0)
    filter_output = np.abs(ifft(fft(xa*np.exp(-1j*phase_tx))*np.conj(fft(np.exp(-1j*phase_tx))) ))
    filter_output = filter_output[:int((N*K)/n_cycle)]
    
    max_idx = np.argmax(np.abs(filter_output))
    tau = ta[max_idx]
    
    if optimize:
        mixed_signal_normalized = mixed_signal / np.sqrt(np.mean(np.square(np.abs(mixed_signal))))
        tau_ = tau * 10e6
        t = np.arange(0, 2*T*n_cycle, 1/sample_rate)[:N]
        phase_tx = fmcw_meas.generate_phase(t, 0, 0)
        def obj_func(tau_):
            phase_rx = fmcw_meas.generate_phase(t, tau_/10e6*3e8/2, 0)
            obj_value = -np.abs(np.sum(mixed_signal_normalized * np.exp(-1j*(phase_tx-phase_rx))))
            return obj_value
        res = minimize_scalar(obj_func, 
                        bracket=(tau_-1/sample_rate/K*10e6, tau_, tau_+1/sample_rate/K*10e6),
                        method="Brent")
        tau = res.x / 10e6
        tau = np.mod(tau, 2*T)
        
    return tau

class MathedFilterDelayEstimator(Estimator):
    def __init__(self, meas_prop: FMCWMeasurementProperties, ignore_quadrature=False,
                 optimize: bool=False):
        self.meas_prop = copy.deepcopy(meas_prop)
        self.ignore_quadrature = ignore_quadrature
        if self.ignore_quadrature:
            self.meas_prop.complex_available = False
        self.test_flag = False
        self.optimize = optimize

    def estimate(self, t: np.ndarray, mixed_signal: np.ndarray, 
                 second_output: np.ndarray|None=None):
        t_hat = matched_filter(self.meas_prop, mixed_signal, optimize=self.optimize)
        d_hat = t_hat * 3e8 / 2
        x_hat = np.array([d_hat,0])
        return x_hat
    
    def set_test_flag(self, test: bool):
        self.test_flag = test

def matched_filter2d(meas_prop: FMCWMeasurementProperties, 
                   mixed_signal: np.ndarray,
                   optimize: bool=False):
    fmcw_meas = fmcw_sys.FMCWMeasurement(meas_prop)
    sample_rate = meas_prop.get_sample_rate()
    bandwidth = meas_prop.get_bandwidth()
    if bandwidth > sample_rate/2:
        K = int(np.ceil(bandwidth/sample_rate*2))
    else:
        K = 1
    T = meas_prop.get_chirp_length()
    # N = len(mixed_signal)
    # mixed_signal = np.pad(mixed_signal, (0, N), mode='constant', constant_values=0)
    n_cycle = int(np.ceil( len(mixed_signal) / (2*T*sample_rate) ))
    ta = np.arange(0, 2*T*n_cycle, 1/sample_rate/K)
    xa = np.zeros((len(ta)), dtype=mixed_signal.dtype)
    N = len(mixed_signal)
    N = min(len(mixed_signal), len(xa[0:N*K:K]))
    xa[0:N*K:K] = mixed_signal[:N]
    phase_tx = fmcw_meas.generate_phase(ta, 0, 0)
    z1 = fft(xa*np.exp(-1j*phase_tx))
    z2 = np.conj(fft(np.exp(-1j*phase_tx)))
    # Z2 = circulant(z2)[:,:N]
    Z2 = np.zeros((len(z2), N), dtype=z2.dtype)
    for i in range(N):
        if i < (N+1)//2:
            Z2[:,i] = np.roll(z2, i)
        else:
            Z2[:,i] = np.roll(z2, i-N)
        # f = fftfreq(M*N, 1/sample_rate)[i]
        # phase_rx = fmcw_meas.generate_phase(ta, 0, f*meas_prop.lambd_c/2)
        # Z2[:,i] = np.conj(fft(np.exp(-1j*phase_rx)))

    filter_output = ifft( np.expand_dims(z1,1) *  Z2, axis=0  )
    filter_output = ifft( filter_output, axis=1 )
    filter_output = np.pad(filter_output, ((0,0),(0, N)), mode="constant", constant_values=0)
    filter_output = np.abs(fft( filter_output, axis=1 ))
    filter_output = filter_output[:int((N*K)/n_cycle)]
    
    max_idx = np.argmax(filter_output.flatten())
    t_idx, f_idx = np.unravel_index(max_idx, filter_output.shape)
    tau = ta[t_idx]
    dop = fftfreq(filter_output.shape[1], 1/sample_rate)[f_idx]
    
    if optimize:
        lambd_c = meas_prop.lambd_c
        mixed_signal_normalized = mixed_signal / np.sqrt(np.mean(np.square(np.abs(mixed_signal))))
        tau_ = tau * 10e6
        dop_ = dop / sample_rate * 100
        t = np.arange(0, 2*T*n_cycle, 1/sample_rate)[:N]
        phase_tx = fmcw_meas.generate_phase(t, 0, 0)
        def obj_func(x):
            phase_rx = fmcw_meas.generate_phase(t, x[0]/10e6*3e8/2, x[1]*sample_rate/10*lambd_c/2)
            obj_value = -np.abs(np.sum(mixed_signal_normalized * np.exp(-1j*(phase_tx-phase_rx))))
            return obj_value
        res = minimize(obj_func, [tau_, dop_],
                       bounds=( (tau_-1/sample_rate/K*10e6, tau_+1/sample_rate/K*10e6), (dop_-10/N, dop_+10/N) ), 
                        method="Nelder-Mead")
        tau = res.x[0] / 10e6
        dop = res.x[1] * sample_rate / 100
        tau = np.mod(tau, 2*T)
        dop = np.mod(dop+sample_rate/2, sample_rate) - sample_rate / 2
        
    return tau, dop

class MathedFilterDelayEstimator2D(Estimator):
    def __init__(self, meas_prop: FMCWMeasurementProperties, ignore_quadrature=False,
                 optimize: bool=False):
        self.meas_prop = copy.deepcopy(meas_prop)
        self.ignore_quadrature = ignore_quadrature
        if self.ignore_quadrature:
            self.meas_prop.complex_available = False
        self.test_flag = False
        self.optimize = optimize

    def estimate(self, t: np.ndarray, mixed_signal: np.ndarray, 
                 second_output: np.ndarray|None=None):
        t_hat, f_hat = matched_filter2d(self.meas_prop, mixed_signal, optimize=self.optimize)
        d_hat = t_hat * 3e8 / 2
        lambd_c = self.meas_prop.lambd_c
        v_hat = f_hat * lambd_c / 2
        x_hat = np.array([d_hat,v_hat])
        return x_hat
    
    def set_test_flag(self, test: bool):
        self.test_flag = test
