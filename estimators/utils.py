
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(this_dir, '..'))
# sys.path.append(os.path.join(this_dir, '..', '..'))

import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft, ifft
import functools
import copy
from typing import Sequence

import fmcw_sys as utils
from fmcw_sys import FMCWMeasurement, FMCWMeasurementProperties
import time_freq
import freq_esti
from optimizer import ObjectiveFunction, Scheduler

MACHINE_EPSILON = np.finfo(np.double).eps

def matched_filter(meas_prop: FMCWMeasurementProperties, t: np.ndarray, mixed_signal: np.ndarray):
    fmcw_meas = utils.FMCWMeasurement(meas_prop)
    sample_rate = meas_prop.get_sample_rate()
    T = meas_prop.get_chirp_length()
    ta = np.arange(0, 2*T, 1/sample_rate)
    xa = np.zeros((len(ta)), dtype=mixed_signal.dtype)
    N = len(mixed_signal)
    xa[:N] = mixed_signal
    phase_tx = fmcw_meas.generate_phase(ta, 0, 0)
    if meas_prop.is_complex_available():
        filter_output = np.real(ifft(fft(xa*np.exp(-1j*phase_tx))*np.conj(fft(np.exp(-1j*phase_tx))) ))
        filter_output = filter_output[:N]
    else:
        filter_output = np.real(ifft(fft(xa*np.exp(-1j*phase_tx))*np.conj(fft(np.exp(-1j*phase_tx))) ) + ifft(fft(xa*np.exp(1j*phase_tx))*np.conj(fft(np.exp(1j*phase_tx))) ))
        filter_output = filter_output[:N]
    return filter_output

def estimate_if(t, y, method='polar_discriminator', mirror=False):
    if method == 'polar_discriminator':
        if y.dtype != np.complex128:
            y = hilbert(y)
        z = np.angle(y[1:] * np.conj(y[:-1])) / (2*np.pi)
        # if y.dtype != np.complex128:
        #     z = utils.mirror(z, 0, 0.5)
        if mirror:
            z = utils.mirror(z, 0, 0.5)
        tt = (t[1:] + t[:-1]) * 0.5
        return tt, z
    elif method == 'reassignment':
        window_length = 17
        hop_length = window_length // 4
        ht = np.arange(-window_length//2, window_length-window_length//2)
        h = np.cos(np.pi*ht/window_length)**2
        dh = 2*np.cos(np.pi*ht/window_length)*(-np.sin(np.pi*ht/window_length))*np.pi/window_length
        # f = np.arange(0, 0.5-0.5/len(h), 1/len(h))
        # sh = time_freq.short_time_fourier_transform_window(y, h, hop_length)[:(len(h)-len(h)//2),:]
        # sdh = time_freq.short_time_fourier_transform_window(y, dh, hop_length)[:(len(h)-len(h)//2),:]
        f = np.fft.fftfreq(len(h), 1)
        sh = time_freq.short_time_fourier_transform_window(y, h, hop_length)
        sdh = time_freq.short_time_fourier_transform_window(y, dh, hop_length)
        maxidx = np.argmax(np.abs(sh), axis=0)
        f_argmax = f[maxidx]
        rf_argmax = f_argmax-np.imag(sdh[maxidx,np.arange(sdh.shape[1])]/sh[maxidx,np.arange(sh.shape[1])]/2/np.pi)
        tt = 0.5 * (t[:(-window_length+1):hop_length] + t[(window_length-1)::hop_length])
        assert len(tt) == sh.shape[1]
        return tt, rf_argmax
    elif method == 'stft':
        window_length = 16
        hop_length = window_length // 4
        ht = np.arange(-window_length//2, window_length-window_length//2)
        h = np.cos(np.pi*ht/window_length)**2
        f = np.fft.fftfreq(len(h), 1)
        sh = time_freq.short_time_fourier_transform_window(y, h, hop_length)
        maxidx = np.argmax(np.abs(sh), axis=0)
        f_argmax = f[maxidx]
        tt = 0.5 * (t[:(-window_length+1):hop_length] + t[(window_length-1)::hop_length])
        assert len(tt) == sh.shape[1]
        return tt, f_argmax
    else:
        raise RuntimeError('unknown method')                          

class IFGenerator():
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        pass

    def __call__(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                 compute_deriv: bool=False, normalize_freq: bool=True, aliased: bool=True):
        raise NotImplementedError()
    
    def get_mixed_if(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                     compute_deriv: bool=False, normalize_freq: bool=True, aliased: bool=True):
        raise NotImplementedError()

    def get_reference_if(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_deriv: bool=False, normalize_freq: bool=True):
        raise NotImplementedError()
    
class TriangularIFGenerator(IFGenerator):
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        if meas_prop.get_modulation_type() != 'triangle':
            raise ValueError('measurement property is not consistent with the class')
        self.meas_prop = copy.deepcopy(meas_prop)
        self.sample_rate = meas_prop.get_sample_rate()
        self.Tchirp = meas_prop.get_chirp_length()
        self.bandwidth = meas_prop.get_bandwidth()
        self.linewidth = meas_prop.get_linewidth()
        self.lambd_c = meas_prop.get_carrier_wavelength()
    
    def get_mixed_if(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                     compute_deriv: bool=False, normalize_freq: bool=True, aliased: bool=True):
        
        d = distance
        v = velocity

        if isinstance(d , Sequence):
            d = np.array(d)
        if isinstance(v, Sequence):
            v = np.array(v)
        if isinstance(d, np.ndarray):
            d = np.expand_dims(d, 1)
        if isinstance(v, np.ndarray):
            v = np.expand_dims(v, 1)

        doppler_shift = v * 2 / self.lambd_c
        delay = d * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        triangle_if_tx = gamma*utils.mirror(t, 0, self.Tchirp)
        t_mirrored, mirror1_deriv = utils.mirror_deriv(t-delay, 0, self.Tchirp)
        triangle_if_rx = gamma*t_mirrored - doppler_shift

        if aliased:
            if self.meas_prop.is_complex_available():
                mixed_if = utils.wrap(triangle_if_tx - triangle_if_rx, -self.sample_rate/2, self.sample_rate/2)
                aliased_deriv = 1
            else:
                mixed_if, aliased_deriv = utils.mirror_deriv(triangle_if_tx - triangle_if_rx, 0, self.sample_rate/2)
        else:
            aliased_deriv = 1

        if compute_deriv:
            d_deriv = aliased_deriv * gamma * mirror1_deriv * 2 / 3e8
            v_deriv = aliased_deriv * 2 / self.lambd_c
            if normalize_freq:
                mixed_if /= self.sample_rate
                d_deriv /= self.sample_rate
                v_deriv /= self.sample_rate
            return_vals =  (mixed_if, d_deriv, v_deriv)
        elif normalize_freq:
            mixed_if /= self.sample_rate
            return_vals = mixed_if
        else:
            return_vals = mixed_if

        return return_vals

    def get_reference_if(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_deriv: bool=False, normalize_freq: bool=True):
        
        sample_rate = self.meas_prop.get_sample_rate()
        doppler_shift = velocity * 2 / self.lambd_c
        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        triangle_if_tx = gamma*utils.mirror(t, 0, self.Tchirp)
        t_mirrored, mirror1_deriv = utils.mirror_deriv(t-delay, 0, self.Tchirp)
        triangle_if_rx = gamma*t_mirrored - doppler_shift

        if normalize_freq:
            triangle_if_rx  /= sample_rate

        return triangle_if_rx
    
    def __call__(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                 compute_deriv: bool=False, normalize_freq: bool=True, aliased: bool=True):
        return self.get_mixed_if(t=t, distance=distance, velocity=velocity, 
                                 compute_deriv=compute_deriv, normalize_freq=normalize_freq, aliased=aliased)
    
def get_triangular_if(meas_prop: FMCWMeasurementProperties, t, normalize_freq=True):
    assert meas_prop.get_modulation_type() == 'triangle'
    sample_rate = meas_prop.get_sample_rate()
    Tchirp = meas_prop.get_chirp_length()
    bandwidth = meas_prop.get_bandwidth()
    linewidth = meas_prop.get_linewidth()
    lambd_c = meas_prop.get_carrier_wavelength()

    gamma = bandwidth / Tchirp
    triangle_if_tx = gamma*utils.mirror(t, 0, Tchirp)
    
    if normalize_freq:
        triangle_if_tx /= sample_rate

    return triangle_if_tx


def get_triangular_aliased_mixed_if(meas_prop: FMCWMeasurementProperties, 
                                     t, d: float|Sequence|np.ndarray, v: float|Sequence|np.ndarray, compute_deriv=False, normalize_freq=True):
                                     
    assert meas_prop.get_modulation_type() == 'triangle'
    if isinstance(d , Sequence):
        d = np.array(d)
    if isinstance(v, Sequence):
        v = np.array(v)
    if isinstance(d, np.ndarray):
        d = np.expand_dims(d, 1)
    if isinstance(v, np.ndarray):
        v = np.expand_dims(v, 1)

    sample_rate = meas_prop.get_sample_rate()
    Tchirp = meas_prop.get_chirp_length()
    bandwidth = meas_prop.get_bandwidth()
    linewidth = meas_prop.get_linewidth()
    lambd_c = meas_prop.get_carrier_wavelength()
    doppler_shift = v * 2 / lambd_c
    delay = d * 2 / 3e8
    gamma = bandwidth / Tchirp
    # Tend = np.ceil(t[-1]/Tchirp)*Tchirp
    # tf_func = np.zeros((len(t)))
    # sign = 1

    triangle_if_tx = gamma*utils.mirror(t, 0, Tchirp)
    t_mirrored, mirror1_deriv = utils.mirror_deriv(t-delay, 0, Tchirp)
    triangle_if_rx = gamma*t_mirrored - doppler_shift
    if meas_prop.is_complex_available():
        mixed_if = utils.wrap(triangle_if_tx - triangle_if_rx, -sample_rate/2, sample_rate/2)
    else:
        mixed_if, mirror2_deriv = utils.mirror_deriv(triangle_if_tx - triangle_if_rx, 0, sample_rate/2)

    if compute_deriv:
        if meas_prop.is_complex_available():
            d_deriv = gamma * mirror1_deriv * 2 / 3e8
            v_deriv = 2 / lambd_c
        else:
            d_deriv = mirror2_deriv * gamma * mirror1_deriv * 2 / 3e8
            v_deriv = mirror2_deriv * 2 / lambd_c
        if normalize_freq:
            mixed_if /= sample_rate
            d_deriv /= sample_rate
            v_deriv /= sample_rate
        return mixed_if, d_deriv, v_deriv
    
    if normalize_freq:
        mixed_if /= sample_rate
    return mixed_if

class SinusoidalIFGenerator(IFGenerator):
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        if meas_prop.get_modulation_type() != 'sinusoidal':
            raise ValueError('measurement property is not consistent with the class')
        self.meas_prop = copy.deepcopy(meas_prop)
        self.sample_rate = meas_prop.get_sample_rate()
        self.Tchirp = meas_prop.get_chirp_length()
        self.bandwidth = meas_prop.get_bandwidth()
        self.linewidth = meas_prop.get_linewidth()
        self.lambd_c = meas_prop.get_carrier_wavelength()
    
    def get_mixed_if(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                     compute_deriv: bool=False, normalize_freq: bool=True, aliased: bool=True):
        
        d = distance
        v = velocity

        if isinstance(d , Sequence):
            d = np.array(d)
        if isinstance(v, Sequence):
            v = np.array(v)
        if isinstance(d, np.ndarray):
            d = np.expand_dims(d, 1)
        if isinstance(v, np.ndarray):
            v = np.expand_dims(v, 1)

        doppler_shift = v * 2 / self.lambd_c
        delay = d * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        sin_if_tx = - 0.5 * self.bandwidth * np.cos(np.pi/self.Tchirp*t)
        sin_if_rx = - 0.5 * self.bandwidth * np.cos(np.pi/self.Tchirp*(t-delay)) - doppler_shift

        if aliased:
            if self.meas_prop.is_complex_available():
                mixed_if = utils.wrap(sin_if_tx - sin_if_rx, -self.sample_rate/2, self.sample_rate/2)
                aliased_deriv = 1
            else:
                mixed_if, aliased_deriv = utils.mirror_deriv(sin_if_tx - sin_if_rx, 0, self.sample_rate/2)
        else:
            aliased_deriv = 1

        if compute_deriv:
            d_deriv = aliased_deriv * 0.5*gamma*np.pi*np.sin(np.pi/self.Tchirp*(t-delay)) * 2 / 3e8
            v_deriv = aliased_deriv * 2 / self.lambd_c
            if normalize_freq:
                mixed_if /= self.sample_rate
                d_deriv /= self.sample_rate
                v_deriv /= self.sample_rate
            return_vals =  (mixed_if, d_deriv, v_deriv)
        elif normalize_freq:
            mixed_if /= self.sample_rate
            return_vals = mixed_if
        else:
            return_vals = mixed_if

        return return_vals

    def get_reference_if(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_deriv: bool=False, normalize_freq: bool=True):
        
        doppler_shift = velocity * 2 / self.lambd_c
        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        sin_if_rx = - 0.5 * self.bandwidth * np.cos(np.pi/self.Tchirp*(t-delay)) - doppler_shift
        
        if normalize_freq:
            sin_if_rx /= self.sample_rate

        return sin_if_rx
    
    def __call__(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                 compute_deriv: bool=False, normalize_freq: bool=True, aliased: bool=True):
        return self.get_mixed_if(t=t, distance=distance, velocity=velocity, 
                                 compute_deriv=compute_deriv, normalize_freq=normalize_freq, aliased=aliased)
    

def get_sinusoidal_if(meas_prop: FMCWMeasurementProperties, t, normalize_freq=True):

    assert meas_prop.get_modulation_type() == 'sinusoidal'

    sample_rate = meas_prop.get_sample_rate()
    Tchirp = meas_prop.get_chirp_length()
    bandwidth = meas_prop.get_bandwidth()
    linewidth = meas_prop.get_linewidth()
    lambd_c = meas_prop.get_carrier_wavelength()

    gamma = bandwidth / Tchirp
    sin_if_tx = - 0.5 * bandwidth * np.cos(np.pi/Tchirp*t)
    
    if normalize_freq:
        sin_if_tx /= sample_rate

    return sin_if_tx

def get_sinusoidal_aliased_mixed_if(meas_prop: FMCWMeasurementProperties,
                                       t, d: float|Sequence|np.ndarray, v: float|Sequence|np.ndarray, compute_deriv=False, normalize_freq=True):
    
    assert meas_prop.get_modulation_type() == 'sinusoidal'

    if isinstance(d , Sequence):
        d = np.array(d)
    if isinstance(v, Sequence):
        v = np.array(v)
    if isinstance(d, np.ndarray):
        d = np.expand_dims(d, 1)
    if isinstance(v, np.ndarray):
        v = np.expand_dims(v, 1)

    sample_rate = meas_prop.get_sample_rate()
    Tchirp = meas_prop.get_chirp_length()
    bandwidth = meas_prop.get_bandwidth()
    linewidth = meas_prop.get_linewidth()
    lambd_c = meas_prop.get_carrier_wavelength()

    doppler_shift = v * 2 / lambd_c
    delay = d * 2 / 3e8
    gamma = bandwidth / Tchirp
    sin_if_tx = - 0.5 * bandwidth * np.cos(np.pi/Tchirp*t)
    sin_if_rx = - 0.5 * bandwidth * np.cos(np.pi/Tchirp*(t-delay)) - doppler_shift

    if meas_prop.is_complex_available():
        mixed_if = utils.wrap(sin_if_tx - sin_if_rx, -sample_rate/2, sample_rate/2)
    else:
        mixed_if, mirror2_deriv = utils.mirror_deriv(sin_if_tx - sin_if_rx, 0, sample_rate/2)

    if compute_deriv:
        if meas_prop.is_complex_available():
            d_deriv = 0.5*gamma*np.pi*np.sin(np.pi/Tchirp*(t-delay)) * 2 / 3e8
            v_deriv = 2 / lambd_c
        else:
            d_deriv = mirror2_deriv * 0.5*gamma*np.pi*np.sin(np.pi/Tchirp*(t-delay)) * 2 / 3e8
            v_deriv = mirror2_deriv * 2 / lambd_c
        if normalize_freq:
            mixed_if /= sample_rate
            d_deriv /= sample_rate
            v_deriv /= sample_rate
        return mixed_if, d_deriv, v_deriv
    
    if normalize_freq:
        mixed_if /= sample_rate
    return mixed_if

def get_trisin_aliased_mixed_if(meas_prop:FMCWMeasurementProperties, 
                                       t, d: float|Sequence|np.ndarray, v: float|Sequence|np.ndarray, compute_deriv=False, normalize_freq=True):

    assert meas_prop.get_modulation_type() == 'mixed'

    if isinstance(d , Sequence):
        d = np.array(d)
    if isinstance(v, Sequence):
        v = np.array(v)
    if isinstance(d, np.ndarray):
        d = np.expand_dims(d, 1)
    if isinstance(v, np.ndarray):
        v = np.expand_dims(v, 1)

    sample_rate = meas_prop.get_sample_rate()
    Tchirp = meas_prop.get_chirp_length()
    bandwidth = meas_prop.get_bandwidth()
    linewidth = meas_prop.get_linewidth()
    lambd_c = meas_prop.get_carrier_wavelength()
    mixed_coef = meas_prop.get_mix_coef()

    doppler_shift = v * 2 / lambd_c
    delay = d * 2 / 3e8
    gamma = bandwidth / Tchirp
    Tend = np.ceil(t[-1]/Tchirp)*Tchirp
    tf_func = np.zeros((len(t)))
    sign = 1
    if compute_deriv:
        d_deriv = np.zeros((len(t)))
    for T in np.arange(0, Tend, Tchirp):
        chirp_period = (T <= t)*(t < delay+T)
        single_period = (delay+T <= t)*(t <= T+Tchirp)
        tf_func[chirp_period] = 2*gamma*(t[chirp_period] - T - delay/2)*sign
        tf_func[single_period] = gamma*delay*sign
        if compute_deriv:
            d_deriv[chirp_period] = -1*sign
            d_deriv[single_period] = sign
        sign *= -1
    tf_func = mixed_coef * tf_func + (1-mixed_coef) * (0.5 * bandwidth * ( - np.cos(np.pi/T*t) + np.cos(np.pi/T*(t - delay)) )) + doppler_shift
    if compute_deriv:
        tf_func, mirror_deriv = utils.mirror_deriv(tf_func, 0, sample_rate/2)
        d_deriv = mirror_deriv * (mixed_coef * d_deriv * gamma + (1-mixed_coef) * 0.5*gamma*np.pi*np.sin(np.pi/T*(t-delay))) * 2 / 3e8
        v_deriv = mirror_deriv * 2 / lambd_c
        if normalize_freq:
            tf_func /= sample_rate
            d_deriv /= sample_rate
            v_deriv /= sample_rate
        return tf_func, d_deriv, v_deriv
    tf_func = utils.mirror(tf_func, 0, sample_rate/2)
    if normalize_freq:
        tf_func /= sample_rate
    return tf_func

def _get_polynomial(t, coefs):
    x = np.zeros_like(t)
    for i, c in enumerate(coefs):
        x = x + t**(i+1)*c
    return x

def _get_polynomial_derivative(t, coefs, lowest_first=True):
    x = np.zeros_like(t)
    for i, c in enumerate(coefs):
        x = x + (i+1)*t**(i)*c
    return x

def compute_polynomial_derivative(t, coefs):
    x = np.zeros_like(t)
    for i, c in enumerate(coefs[-2::-1]):
        x = x + (i+1)*(t**(i))*c
    return x

def _get_polynomial_derivative2(t, coefs):
    x = np.zeros_like(t)
    for i, c in enumerate(coefs[1:]):
        x = x + (i+2)*(i+1)*(t**(i))*c
    return x

def get_polynomial_aliased_mixed_if(meas_prop: FMCWMeasurementProperties,
                                       t, d: float|Sequence|np.ndarray, v: float|Sequence|np.ndarray, compute_deriv=False, normalize_freq=True):

    assert meas_prop.get_modulation_type() == 'polynomial'

    if isinstance(d , Sequence):
        d = np.array(d)
    if isinstance(v, Sequence):
        v = np.array(v)
    if isinstance(d, np.ndarray):
        d = np.expand_dims(d, 1)
    if isinstance(v, np.ndarray):
        v = np.expand_dims(v, 1)

    sample_rate = meas_prop.get_sample_rate()
    Tchirp = meas_prop.get_chirp_length()
    bandwidth = meas_prop.get_bandwidth()
    linewidth = meas_prop.get_linewidth()
    lambd_c = meas_prop.get_carrier_wavelength()
    up_coefs, down_coefs = meas_prop.get_poly_coefs()

    doppler_shift = v * 2 / lambd_c
    delay = d * 2 / 3e8
    Tend = np.ceil(t[-1]/Tchirp)*Tchirp
    tf_func = np.zeros((len(t)))
    if compute_deriv:
        d_deriv = np.zeros((len(t)))
    for T in np.arange(0, Tend, 2*Tchirp):
        interval1 = (T <= t)*(t < T+delay)
        interval2 = (delay+T <= t)*(t <= T+Tchirp)
        interval3 = (T+Tchirp <= t)*(t <= T+Tchirp+delay)
        interval4 = (T+Tchirp+delay <= t)*(t <= T+Tchirp+delay)

        tf_func[interval1] = _get_polynomial_derivative(t[interval1]-T, up_coefs) + _get_polynomial_derivative(t[interval1]-T+Tchirp-delay, down_coefs) + doppler_shift
        tf_func[interval2] = _get_polynomial_derivative(t[interval2]-T, up_coefs) - _get_polynomial_derivative(t[interval2]-T-delay, up_coefs) + doppler_shift
        tf_func[interval3] = - _get_polynomial_derivative(t[interval3]-T-Tchirp, down_coefs) - _get_polynomial_derivative(t[interval3]-T-delay, up_coefs) + doppler_shift
        tf_func[interval4] = - _get_polynomial_derivative(t[interval4]-T-Tchirp, down_coefs) + _get_polynomial_derivative(t[interval4]-T-Tchirp-delay, down_coefs) + doppler_shift

        if compute_deriv:
            d_deriv[interval1] = - _get_polynomial_derivative2(t[interval1]-T+Tchirp-delay, down_coefs)
            d_deriv[interval2] = _get_polynomial_derivative2(t[interval2]-T-delay, up_coefs)
            d_deriv[interval3] = _get_polynomial_derivative2(t[interval3]-T-delay, up_coefs)
            d_deriv[interval4] = - _get_polynomial_derivative2(t[interval4]-T-Tchirp-delay, down_coefs)

    if compute_deriv:
        tf_func, mirror_deriv = utils.mirror_deriv(tf_func, 0, sample_rate/2)
        d_deriv = mirror_deriv * d_deriv * 2 / 3e8
        v_deriv = mirror_deriv * 2 / lambd_c
        if normalize_freq:
            tf_func /= sample_rate
            d_deriv /= sample_rate
            v_deriv /= sample_rate
        return tf_func, d_deriv, v_deriv
    tf_func = utils.mirror(tf_func, 0, sample_rate/2)
    if normalize_freq:
        tf_func /= sample_rate
    return tf_func

def get_if_generator(meas_prop: FMCWMeasurementProperties):
    modulation = meas_prop.get_modulation_type()
    if modulation == 'triangle':
        return TriangularIFGenerator(meas_prop)
    elif modulation == 'sinusoidal':
        return SinusoidalIFGenerator(meas_prop)
    elif modulation == 'mixed':
        raise NotImplementedError()
        # return get_trisin_aliased_mixed_if
    elif modulation == 'polynomial':
        raise NotImplementedError()
        # return get_polynomial_aliased_mixed_if
    elif modulation == 'triangle_phase':
        raise NotImplementedError()
    else:
        raise ValueError('modulation type \"{:s}\" not found'.format(modulation))
    
class GridGenerator():
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        self.meas_prop = copy.deepcopy(meas_prop)

    def generate_lattice_grid(self):
        
        d_range, v_range = self.meas_prop.get_range()

        B = self.meas_prop.get_bandwidth()
        sample_rate = self.meas_prop.get_sample_rate()
        T = self.meas_prop.get_chirp_length()
        
        if self.meas_prop.get_modulation_type() == 'triangle':
            d_interval = sample_rate/2/B*T*2*3e8/2
            if not self.meas_prop.is_complex_available():
                d_interval = d_interval / 2
        elif self.meas_prop.get_modulation_type() == 'sinusoidal':
            d_interval = np.arcsin(sample_rate/2/B)*T/np.pi*2*3e8/2
            if not self.meas_prop.is_complex_available():
                d_interval = d_interval / 2
        else:
            raise ValueError('lattice generation only supported for triangular and sinusoidal modulation')

        precaution_rate = 0.9
        d_interval = d_interval * precaution_rate
        
        if not self.meas_prop.boundary_constraint:
            d_grid = np.arange(d_range[0], d_range[-1]-MACHINE_EPSILON, d_interval)
        else:
            d_cuts = np.concatenate([np.arange(d_range[0], d_range[-1]-MACHINE_EPSILON, d_interval), [d_range[-1]]])
            d_grid = (d_cuts[1:]+d_cuts[:-1])/2

        if self.meas_prop.is_zero_velocity():
            dv_grid = np.stack([d_grid, np.zeros_like(d_grid)], axis=1)
        else:
            max_v = self.meas_prop.get_max_v()
            if self.meas_prop.is_complex_available():
                if not self.meas_prop.boundary_constraint:
                    d_grid_shift = np.arange(d_range[0]+d_interval/2, d_range[-1]-MACHINE_EPSILON, d_interval)
                    dv_grid1 = np.stack([d_grid, np.zeros_like(d_grid)], axis=1)
                    dv_grid2 = np.stack([d_grid_shift, np.ones_like(d_grid_shift)*max_v], axis=1)
                    dv_grid = np.concatenate([dv_grid1, dv_grid2], axis=0)
                else:
                    raise NotImplementedError('lattice generation only supported for no boundary constraint')
            else:
                if not self.meas_prop.boundary_constraint:
                    d_grid_shift = np.arange(d_range[0]+d_interval/2, d_range[-1]-MACHINE_EPSILON, d_interval)
                    dv_grid1 = np.stack([d_grid, np.zeros_like(d_grid)], axis=1)
                    dv_grid2 = np.stack([d_grid_shift, np.ones_like(d_grid_shift)*max_v/2], axis=1)
                    dv_grid3 = np.stack([d_grid_shift, -np.ones_like(d_grid_shift)*max_v/2], axis=1)
                    dv_grid4 = np.stack([d_grid_shift, np.ones_like(d_grid_shift)*max_v], axis=1)
                    dv_grid = np.concatenate([dv_grid1, dv_grid2, dv_grid3, dv_grid4], axis=0)
                else:
                    raise NotImplementedError('lattice generation only supported for no boundary constraint')
        return dv_grid
        
    def generate_uniform_grid(self, d_interval: float|None=None, v_interval: float|None=None, 
                              num_d: int|None=None, num_v: int|None=None, 
                              d_range: Sequence|None=None, v_range: Sequence|None=None):

        # max_d = self.meas_prop.get_max_d()
        # if num_d is None:
        #     if self.meas_prop.get_modulation_type() == 'triangle':
        #         B = self.meas_prop.get_bandwidth()
        #         sample_rate = self.meas_prop.get_sample_rate()
        #         T = self.meas_prop.get_chirp_length()
        #         interval = sample_rate/2/B*T*3e8/2 * 1.8
        #         num_d = int(np.ceil(max_d/interval))
        #         if not self.meas_prop.is_complex_available():
        #             num_d = num_d * 2
        #     elif self.meas_prop.get_modulation_type() == 'sinusoidal':
        #         B = self.meas_prop.get_bandwidth()
        #         sample_rate = self.meas_prop.get_sample_rate()
        #         T = self.meas_prop.get_chirp_length()
        #         interval = np.arcsin(sample_rate/2/B)*T/np.pi*2*3e8/2
        #         num_d = int(np.ceil(max_d/interval))
        #         if not self.meas_prop.is_complex_available():
        #             num_d = num_d * 2
        #     else:
        #         raise ValueError('modulation must be either triangle or sinusoidal for auto resolution decision for uniform grid')

        if d_range is None:
            max_d = self.meas_prop.get_max_d()
            d_range = (0, max_d)

        if (d_interval is None) and (num_d is None):
            B = self.meas_prop.get_bandwidth()
            sample_rate = self.meas_prop.get_sample_rate()
            T = self.meas_prop.get_chirp_length()
            if self.meas_prop.get_modulation_type() == 'triangle':
                d_interval = sample_rate/2/B*T*2*3e8/2*0.9
                if not self.meas_prop.is_complex_available():
                    d_interval = d_interval / 2
            elif self.meas_prop.get_modulation_type() == 'sinusoidal':
                d_interval = np.arcsin(sample_rate/2/B)*T/np.pi*2*3e8/2*0.9
                if not self.meas_prop.is_complex_available():
                    d_interval = d_interval / 2
            else:
                raise ValueError('automatic uniform grid generation only supported for triangular and sinusoidal modulation')
        elif (d_interval is None) and (num_d is not None):
            d_interval = (d_range[1]-d_range[0]) / num_d
        
        d_cut1 = np.arange(d_range[0], d_range[1]-d_interval/2, d_interval)
        d_cut2 = np.arange(d_range[0]+d_interval, d_range[1]+d_interval/2, d_interval)
        d_grid = (d_cut1 + d_cut2) / 2

        if self.meas_prop.is_zero_velocity():
            v_grid = np.array([0])
        else:
            if v_range is None:
                max_v = self.meas_prop.get_max_v()
                v_range = (-max_v, max_v)

            if (v_interval is None) and (num_v is None):
                B = self.meas_prop.get_bandwidth()
                sample_rate = self.meas_prop.get_sample_rate()
                T = self.meas_prop.get_chirp_length()
                if self.meas_prop.get_modulation_type() == 'triangle':
                    v_interval = sample_rate/2/B*T*2*3e8/2*0.9
                    if not self.meas_prop.is_complex_available():
                        v_interval = v_interval / 2
                elif self.meas_prop.get_modulation_type() == 'sinusoidal':
                    v_interval = np.arcsin(sample_rate/2/B)*T/np.pi*2*3e8/2*0.9
                    if not self.meas_prop.is_complex_available():
                        v_interval = v_interval / 2
                else:
                    raise ValueError('automatic uniform grid generation only supported for triangular and sinusoidal modulation')
            elif (v_interval is None) and (num_v is not None):
                v_interval = (v_range[1]-v_range[0]) / num_v
        
            v_cut1 = np.arange(v_range[0], v_range[1]-v_interval/2, v_interval)
            v_cut2 = np.arange(v_range[0]+v_interval, v_range[1]+v_interval/2, v_interval)
            v_grid = (v_cut1 + v_cut2) / 2

        # grid = np.zeros((len(d_grid), len(v_grid), 2))
        # for i, d in enumerate(d_grid):
        #     for j, v in enumerate(v_grid):
        #         grid[i,j,0] = d
        #         grid[i,j,1] = v
        d_grid_2d, v_grid_2d = np.meshgrid(d_grid, v_grid, indexing='ij')
        grid = np.stack([d_grid_2d, v_grid_2d], axis=-1)

        return grid


    def generate_smart_grid(self, t: np.ndarray, signal: np.ndarray, second_output: np.ndarray|None=None, perturb: bool=False, dither_std=None, method: str='lorentzian'):
        
        assert self.meas_prop.modulation == 'triangle'

        if self.meas_prop.is_zero_velocity() and ( (t[-1] + 2/self.meas_prop.sample_rate - t[0])<(self.meas_prop.Tchirp) ):
            raise RuntimeError('not enough observations')
        elif not self.meas_prop.is_zero_velocity() and ( (t[-1] + 2/self.meas_prop.sample_rate - t[0])<(2*self.meas_prop.Tchirp) ):
            raise RuntimeError('not enough observations')
    
        max_d = self.meas_prop.get_max_d()
        max_v = self.meas_prop.get_max_v()
        Tchirp = self.meas_prop.get_chirp_length()
        sample_rate = self.meas_prop.get_sample_rate()
        bandwidth = self.meas_prop.get_bandwidth()
        assume_zero_velocity = self.meas_prop.is_zero_velocity()
        linewidth = self.meas_prop.get_linewidth()
        lambd_c = self.meas_prop.get_carrier_wavelength()
        complex_available = self.meas_prop.is_complex_available()

        d_nyquist = sample_rate / 2 / (bandwidth / Tchirp)  * 3e8 / 2
        v_nyquist = sample_rate / 2 * lambd_c / 2

        if perturb:
            raise NotImplementedError('perturb not implemented yet')

        if ((Tchirp - max_d * 2 / 3e8) * sample_rate) < 10:
            raise RuntimeError('max distance allowed must be shorter than chirp period')
        
        meas_start = max_d * 2 / 3e8
        meas_end = Tchirp
        meas1_idx = (t > meas_start) * (t < meas_end)
        meas2_idx = (t > (Tchirp + meas_start)) * (t < (Tchirp + meas_end))
        if method == 'lorentzian':
            if second_output is not None:
                shot_var = np.var(second_output)
            else:
                shot_var = 0
            res1, success = freq_esti.lorentzian_fitting(t[meas1_idx], signal[meas1_idx], shot_var=shot_var, meas_prop=self.meas_prop)
            if not success:
                return None
            freq1 = res1[0] / sample_rate
            if not assume_zero_velocity:
                res2, success = freq_esti.lorentzian_fitting(t[meas2_idx], signal[meas2_idx], shot_var=shot_var, meas_prop=self.meas_prop)
                if not success:
                    return None
                freq2 = res2[0] / sample_rate
        elif method == 'maxpd_fine':
            res1, success = freq_esti.max_periodogram(signal[meas1_idx], method='fine', meas_prop=self.meas_prop)
            if not success:
                return None
            freq1 = res1
            if not assume_zero_velocity:
                res2, success = freq_esti.max_periodogram(signal[meas2_idx], method='fine', meas_prop=self.meas_prop)
                if not success:
                    return None
                freq2 = res2 / sample_rate
        elif method == 'maxpd_coarse':
            res1, _ = freq_esti.max_periodogram(signal[meas1_idx], method='coarse', meas_prop=self.meas_prop)
            freq1 = res1
            if not assume_zero_velocity:
                res2, _ = freq_esti.max_periodogram(signal[meas2_idx], method='coarse', meas_prop=self.meas_prop)
                freq2 = res2 / sample_rate

        if assume_zero_velocity:
            if complex_available:
                gamma = bandwidth / Tchirp
                k = int(np.floor(max_d / (d_nyquist*2) + 1/2))
                freq1um = freq1 + np.arange(0, k+1)
                grid = np.zeros((len(freq1um), 2))
                grid[:,0] =  freq1um * sample_rate / gamma / 2 * 3e8
            else:
                gamma = bandwidth / Tchirp
                n_unmirror_max = int(np.floor(max_d / d_nyquist))
                freq1um = utils.unmirror(freq1, 0, 0.5, n_min=0, n_max=n_unmirror_max)
                grid = np.zeros((len(freq1um), 2))
                grid[:,0] =  freq1um * sample_rate / gamma / 2 * 3e8
            if perturb:
                grid[:,0] = utils.mirror(grid[:,0] + np.random.randn() * dither_std[0], 0, max_d)
        else:
            if complex_available:
                gamma = bandwidth / Tchirp
                k = int(np.ceil(max_d / (d_nyquist*2) + 1/2))
                freq1um = freq1 + np.arange(-1, k+1)
                freq2um = freq2 + np.arange(-1, k+1)
                dist_vel_pairs = np.zeros((len(freq1um), len(freq2um), 2))
                dist_vel_pairs[:,:,0] = ( np.reshape(freq1um, (-1,1)) - np.reshape(freq2um, (1,-1)) ) / 2 * sample_rate / gamma / 2 * 3e8
                dist_vel_pairs[:,:,1] = ( np.reshape(freq1um, (-1,1)) + np.reshape(freq2um, (1,-1)) ) / 2 * sample_rate / 2 * lambd_c
                if perturb:
                    dist_vel_pairs[:,:,0] = utils.mirror(dist_vel_pairs[:,:,0] + np.random.randn() * dither_std[0], 0, max_d)
                    dist_vel_pairs[:,:,1] = utils.mirror(dist_vel_pairs[:,:,1] + np.random.randn() * dither_std[1], -max_v, max_v)
                valid_pairs = np.zeros((len(freq1um), len(freq2um)), dtype=bool)
                valid_pairs[(dist_vel_pairs[:,:,0]>0)*(dist_vel_pairs[:,:,0]<max_d)] = True
                valid_pairs[np.abs(dist_vel_pairs[:,:,1])>=max_v] = False
                grid = dist_vel_pairs[valid_pairs]
            else:
                gamma = bandwidth / Tchirp
                n_unmirror_max = int(np.ceil(max_d * 2 / 3e8 * gamma / sample_rate * 2))
                freq1um = utils.unmirror(freq1, 0, 0.5, n_min=-1, n_max=n_unmirror_max)
                freq2um = utils.unmirror(freq2, 0, 0.5, n_min=-1, n_max=n_unmirror_max)
                dist_vel_pairs = np.zeros((len(freq1um), len(freq2um), 2))
                dist_vel_pairs[:,:,0] = ( np.reshape(freq1um, (-1,1)) + np.reshape(freq2um, (1,-1)) ) / 2 * sample_rate / gamma / 2 * 3e8
                dist_vel_pairs[:,:,1] = ( np.reshape(freq1um, (-1,1)) - np.reshape(freq2um, (1,-1)) ) / 2 * sample_rate / 2 * lambd_c
                if perturb:
                    dist_vel_pairs[:,:,0] = utils.mirror(dist_vel_pairs[:,:,0] + np.random.randn() * dither_std[0], 0, max_d)
                    dist_vel_pairs[:,:,1] = utils.mirror(dist_vel_pairs[:,:,1] + np.random.randn() * dither_std[1], -max_v, max_v)
                valid_pairs = np.zeros((len(freq1um), len(freq2um)), dtype=bool)
                valid_pairs[(dist_vel_pairs[:,:,0]>0)*(dist_vel_pairs[:,:,0]<max_d)] = True
                valid_pairs[np.abs(dist_vel_pairs[:,:,1])>=max_v] = False
                grid = dist_vel_pairs[valid_pairs]

        return grid
    
    def generate_smart_grid_sinmod(self, t: np.ndarray, signal: np.ndarray, num_d=40, d_range=None, 
                                   perturb=False, dither_std=None, method='maxpd_coarse'):
        
        assert self.meas_prop.modulation == 'sinusoidal'
        assert self.meas_prop.assume_zero_velocity == False

        Tchirp = self.meas_prop.get_chirp_length()
        signal1 = signal[(t>=0)*(t<Tchirp)]
        signal2 = signal[(t>=Tchirp)*(t<2*Tchirp)]
        if len(signal1) != len(signal2):
            length = np.minimum(len(signal1), len(signal2))
            signal1 = signal1[:length]
            signal2 = signal2[:length]
        
        mixed_signal = signal1 * signal2
        if method == 'maxpd_coarse':
            beatfreq, _ = freq_esti.max_periodogram(mixed_signal, method='coarse')
        elif method == 'maxpd_fine':
            beatfreq, success = freq_esti.max_periodogram(mixed_signal, method='fine')
            if not success:
                return None
            
        lambd_c = self.meas_prop.get_carrier_wavelength()
        sample_rate = self.meas_prop.get_sample_rate()
        beatfreqs = utils.unmirror(beatfreq, 0, 0.5, n_min=-2, n_max=1)
        v_hats = beatfreqs * sample_rate / 4 * lambd_c

        max_d = self.meas_prop.get_max_d()
        if d_range is not None:
            d_range = np.linspace(d_range[0], d_range[1], num_d+2)[1:-1]
        else:
            d_range = np.linspace(0, max_d, num_d+2)[1:-1]

        if perturb and (dither_std is None):
            dither_std = (0.2 * (d_range[1] - d_range[0]), 0)
        grid = np.zeros((len(d_range), 4, 2))
        for i, d in enumerate(d_range):
            for j, v in enumerate(v_hats):
                if perturb:
                    d = utils.mirror(d + np.random.randn() * dither_std[0], d_range[0], d_range[-1])
                grid[i,j,0] = d
                grid[i,j,1] = v
        grid = np.reshape(grid, (-1, 2))

        return grid

class IFLikelihood():
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        self.meas_prop = copy.deepcopy(meas_prop)
    def evaluate_nll(self):
        raise NotImplementedError()
    def compute_nll_gradient(self):
        raise NotImplementedError()
    def nll_grid_search(self):
        raise NotImplementedError()

# class IFMirroredNormal(IFNoiseModel):
#     def __init__(self, meas_prop: FMCWMeasurementProperties):
#         self.meas_prop = copy.deepcopy(meas_prop)
#         self.gen_tf_func = IFGenerator(meas_prop)
#         linewidth = self.meas_prop.get_linewidth()
#         sample_rate = self.meas_prop.get_sample_rate()
#         self.tf_var = linewidth*2/sample_rate/(2*np.pi)
        
#     def evaluate_nll(self, t: np.ndarray, tf_func: np.ndarray, distance: float, velocity: float, eta: float=1e-20):
#         tf_hat = self.gen_tf_func(t, distance, velocity, compute_deriv=False, normalize_freq=True)
#         Z1 = np.exp(-(tf_func-tf_hat)**2/(2*self.tf_var))
#         Z2 = np.exp(-(tf_func+tf_hat)**2/(2*self.tf_var))
#         Z3 = np.exp(-(tf_func-1+tf_hat)**2/(2*self.tf_var))
#         nll = np.sum( -np.log(Z1 + Z2 + Z3 + eta) )
#         return nll
    
#     def compute_nll_graident(self, time: np.ndarray, ifrq: np.ndarray, distance: float, velocity: float, eta: float=1e-20):
#         ifrq_hat, d_deriv, v_deriv = self.gen_tf_func(time, distance, velocity, compute_deriv=True, normalize_freq=True)
#         Z1 = np.exp(-(ifrq-ifrq_hat)**2/(2*self.tf_var))
#         Z2 = np.exp(-(ifrq+ifrq_hat)**2/(2*self.tf_var))
#         Z3 = np.exp(-(ifrq-1+ifrq_hat)**2/(2*self.tf_var))
#         denom = Z1 + Z2 + Z3 + eta
#         # numer = - (Z1 * (ifrq_hat-ifrq) + Z2 * (ifrq_hat+ifrq) + Z3 * (ifrq_hat-1+ifrq)) / self.tf_var
#         numer = (Z1 * (ifrq-ifrq_hat) + Z2 * -(ifrq+ifrq_hat) + Z3 * -(ifrq-1+ifrq_hat)) / self.tf_var
#         tmp = - numer / denom
#         d_deriv = np.sum(tmp * d_deriv)
#         v_deriv = np.sum(tmp * v_deriv)
#         return d_deriv, v_deriv

#     def nll_grid_search(self, t:np.ndarray, tf_func:np.ndarray, grid:np.ndarray, return_grid_val:bool=False):
#         grid_orig_shape = grid.shape
#         grid = np.reshape(grid, (-1, 2))
#         grid_val = np.zeros(len(grid))
#         for i, d_v_pair in enumerate(grid):
#             grid_val[i] = self.evaluate_nll(t, tf_func, d_v_pair[0], d_v_pair[1])
#         idx = np.argmin(grid_val)
#         d_hat = grid[idx, 0]
#         v_hat = grid[idx, 1]
#         if return_grid_val:
#             return d_hat, v_hat, np.reshape(grid_val, grid_orig_shape[:-1])
#         return d_hat, v_hat


class IFWrappedNormalLikelihood(IFLikelihood, ObjectiveFunction):
    def __init__(self, meas_prop: FMCWMeasurementProperties, K=2, eps=1e-20):
        self.meas_prop = copy.deepcopy(meas_prop)
        self.if_generator = get_if_generator(meas_prop)
        self.linewidth = self.meas_prop.get_linewidth()
        self.sample_rate = self.meas_prop.get_sample_rate()
        self.if_var = self.linewidth*2/self.sample_rate/(2*np.pi)
        self.K = K
        self.snr_adjust_polyfit_coefs = np.load(os.path.join(this_dir,'phase_noise','project_normal_diff_var_polyfit.npy'))
        self.snr_adjust_polyfit_func = np.poly1d(self.snr_adjust_polyfit_coefs)
        self.eps = eps
        self.t = None
        self.ifx = None
    
    def snr_adjust_func(self, snr: float|np.ndarray, compute_deriv=False, dB=True):
        if not dB:
            snr_dB = 10*np.log10(snr+1e-20)
        else:
            snr_dB = snr
        if isinstance(snr, np.ndarray):
            adjust_var = np.zeros_like(snr)
            adjust_var[snr_dB>30] = 0
            adjust_var[snr_dB<-20] = 1/12
            adjust_var[(snr_dB>=-20)*(snr_dB<=30)] = self.snr_adjust_polyfit_func(snr_dB[(snr_dB>=-20)*(snr_dB<=30)]) / (2*np.pi)**2
            if compute_deriv:
                adjust_var_deriv = np.ones_like(snr)
                if not dB:
                    adjust_var_deriv = 10 * adjust_var_deriv / (snr*np.log(10))
                adjust_var_deriv[snr_dB>30] = 0
                adjust_var_deriv[snr_dB<-20] = 0
                adjust_var_deriv[(snr_dB>=-20)*(snr_dB<=30)] *= compute_polynomial_derivative(snr_dB[(snr_dB>=-20)*(snr_dB<=30)], self.snr_adjust_polyfit_coefs) / (2*np.pi**2)
                return adjust_var, adjust_var_deriv
            return adjust_var
        else:
            if snr_dB > 30:
                adjust_var = 0
            elif snr_dB < -20:
                adjust_var = 1/12
            else:
                adjust_var = self.snr_adjust_polyfit_func(snr_dB) / (2*np.pi)**2
            if compute_deriv:
                adjust_var_deriv = 1
                if not dB:
                    adjust_var_deriv = adjust_var_deriv / (snr*np.log(10))
                if snr_dB > 30:
                    adjust_var_deriv = 0
                elif snr_dB < -20:
                    adjust_var_deriv = 0
                else:
                    adjust_var_deriv = adjust_var_deriv * compute_polynomial_derivative(snr_dB, self.snr_adjust_polyfit_coefs) / (2*np.pi**2)
                return adjust_var, adjust_var_deriv
            return adjust_var

    def adjust_for_snr(self, snr, dB=True):
        self.if_var = self.linewidth*2/self.sample_rate/(2*np.pi) + self.snr_adjust_func(snr, dB=dB)

    def remove_snr_adjustment(self):
        self.if_var = self.linewidth*2/self.sample_rate/(2*np.pi)
        
    # def evaluate_nll(self, t: np.ndarray, if_y: np.ndarray, distance: float, velocity: float):
    #     if_hat = self.if_generator.get_mixed_if(t, distance, velocity, compute_deriv=False, normalize_freq=True, aliased=True)
    #     if self.meas_prop.is_complex_available() == False:
    #         if_var = self.if_var
    #     else:
    #         if_var = self.if_var
    #     Z = np.zeros((len(t)))
    #     for k in np.arange(-self.K, self.K+1):
    #         Z = Z + np.exp(-(if_y-if_hat+k)**2/(2*if_var))
    #     nll = np.sum( -np.log(Z+self.eps) )
    #     return nll
    
    # def compute_nll_graident(self, t: np.ndarray, if_y: np.ndarray, distance: float, velocity: float):
    #     if_hat, d_deriv, v_deriv = self.if_generator.get_mixed_if(t, distance, velocity, compute_deriv=True, normalize_freq=True, aliased=True)
    #     if self.meas_prop.is_complex_available() == False:
    #         if_var = self.if_var
    #         if_var_deriv = 0
    #     else:
    #         if_var = self.if_var
    #         if_var_deriv = 0
    #     denom = np.zeros((len(t)))
    #     numer = np.zeros((len(t)))
    #     for k in np.arange(-self.K, self.K+1):
    #         Z = np.exp(-(if_y-if_hat+k)**2/(2*if_var))
    #         denom = denom + Z
    #         numer = numer + Z * ( (if_y-if_hat+k)/(if_var) + (if_y-if_hat+k)**2/(2*if_var**2)*if_var_deriv )
    #     tmp = - numer / (denom + self.eps)
    #     d_deriv = np.sum(tmp * d_deriv)
    #     v_deriv = np.sum(tmp * v_deriv)
    #     if self.meas_prop.assume_zero_velocity:
    #         v_deriv = 0
    #     return d_deriv, v_deriv

    # def nll_grid_search(self, t:np.ndarray, if_y:np.ndarray, grid:np.ndarray, return_grid_val:bool=False):
    #     grid_orig_shape = grid.shape
    #     grid = np.reshape(grid, (-1, 2))
    #     grid_val = np.zeros(len(grid))
    #     for i, d_v_pair in enumerate(grid):
    #         grid_val[i] = self.evaluate_nll(t, if_y, d_v_pair[0], d_v_pair[1])
    #     idx = np.argmin(grid_val)
    #     d_hat = grid[idx, 0]
    #     v_hat = grid[idx, 1]
    #     if return_grid_val:
    #         return d_hat, v_hat, np.reshape(grid_val, grid_orig_shape[:-1])
    #     return d_hat, v_hat
    
    def store_observations(self, t: np.ndarray, ifx: np.ndarray):

        assert (np.ndim(t) == 1) and (np.ndim(ifx) == 1)

        if self.meas_prop.is_zero_velocity() and ( (t[-1] + 3/self.sample_rate - t[0])<(self.meas_prop.Tchirp) ):
            raise RuntimeError('not enough observations')
        elif not self.meas_prop.is_zero_velocity() and ( (t[-1] + 3/self.sample_rate - t[0])<(2*self.meas_prop.Tchirp) ):
            raise RuntimeError('not enough observations')
        
        self.t = copy.deepcopy(t)
        self.ifx = copy.deepcopy(ifx)

    def evaluate(self, x: np.ndarray, extra_var: float|None=None):

        if (np.ndim(x) != 2) or (x.shape[-1] != 2):
            raise ValueError("Dimension of x not acceptable")

        if extra_var is None:
            extra_var = 0

        if self.meas_prop.is_zero_velocity():
            if_hat = self.if_generator.get_mixed_if(self.t, x[:,0], 0, compute_deriv=False, normalize_freq=True, aliased=True)
        else:
            if_hat = self.if_generator.get_mixed_if(self.t, x[:,0], x[:,1], compute_deriv=False, normalize_freq=True, aliased=True)

        Z = np.zeros((len(self.t)))
        for k in np.arange(-self.K, self.K+1):
            Z = Z + np.exp(-(self.ifx-if_hat+k)**2/(2*(self.if_var+extra_var)))
        nll = np.sum( -np.log(Z+MACHINE_EPSILON), axis=1) + len(self.t)/2*np.log(2*(self.if_var+extra_var))
        # nll = np.sum( -np.log(Z+self.eps), axis=1) * 2*(self.if_var+extra_var)
        return nll
    
    def compute_gradient(self, x: np.ndarray, extra_var: float|None=None):

        if (np.ndim(x) != 2) or (x.shape[-1] != 2):
            raise ValueError("Dimension of x not acceptable")

        if extra_var is None:
            extra_var = 0

        if self.meas_prop.is_zero_velocity():
            if_hat, d_deriv, v_deriv = self.if_generator.get_mixed_if(self.t, x[:,0], 0, compute_deriv=True, normalize_freq=True, aliased=True)
        else:
            if_hat, d_deriv, v_deriv =self.if_generator.get_mixed_if(self.t, x[:,0], x[:,1], compute_deriv=True, normalize_freq=True, aliased=True)

        denom = np.zeros((len(self.t)))
        numer = np.zeros((len(self.t)))
        for k in np.arange(-self.K, self.K+1):
            Z = np.exp(-(self.ifx-if_hat+k)**2/(2*(self.if_var+extra_var)))
            denom = denom + Z
            numer = numer + Z * ( (self.ifx-if_hat+k)/(self.if_var+extra_var) )
        tmp = - numer / (denom + MACHINE_EPSILON)
        d_deriv = np.sum(tmp * d_deriv, axis=1)
        v_deriv = np.sum(tmp * v_deriv, axis=1)
        if self.meas_prop.is_zero_velocity():
            gradient = np.stack([d_deriv, np.zeros_like(v_deriv)], axis=1)
        else:
            gradient = np.stack([d_deriv, v_deriv], axis=1)
        return gradient

    def evaluate_and_compute_gradient(self, x: np.ndarray, extra_var: float|None=None):

        if (np.ndim(x) != 2) or (x.shape[-1] != 2):
            raise ValueError("Dimension of x not acceptable")
        
        if extra_var is None:
            extra_var = 0

        if self.meas_prop.assume_zero_velocity:
            if_hat, d_deriv, v_deriv = self.if_generator.get_mixed_if(self.t, x[:,0], 0, compute_deriv=True, normalize_freq=True, aliased=True)
        else:
            if_hat, d_deriv, v_deriv = self.if_generator.get_mixed_if(self.t, x[:,0], x[:,1], compute_deriv=True, normalize_freq=True, aliased=True)
        
        Z = np.zeros((len(self.t)))
        denom = np.zeros((len(self.t)))
        numer = np.zeros((len(self.t)))
        for k in np.arange(-self.K, self.K+1):
            Zk = np.exp(-(self.ifx-if_hat+k)**2/(2*(self.if_var+extra_var)))
            Z = Z + Zk
            denom = denom + Zk
            numer = numer + Zk * ( (self.ifx-if_hat+k)/(self.if_var+extra_var) )
        nll = - np.sum( np.log(Z+MACHINE_EPSILON), axis=1) + len(self.t)/2*np.log(2*(self.if_var+extra_var))
        # nll = - np.sum( np.log(Z+self.eps), axis=1) * 2*(self.if_var+extra_var)
        tmp = - numer / (denom + MACHINE_EPSILON)
        d_deriv = np.sum(tmp * d_deriv, axis=1)
        v_deriv = np.sum(tmp * v_deriv, axis=1)
        if self.meas_prop.is_zero_velocity():
            gradient = np.stack([d_deriv, np.zeros_like(v_deriv)], axis=1)
        else:
            gradient = np.stack([d_deriv, v_deriv], axis=1)
        return nll, gradient
    
    def create_scheduler(self):
        # return ExtraVarScheduler(self.if_var, init_var=0.01, m=300)
        return ExtraVarScheduler(self.if_var, init_var=0.01, m=30)

class ExtraVarScheduler(Scheduler):
    def __init__(self, correct_var: float, init_var: float=0.01, m: int=70):
    
        self._curr_x = None
        self._correct_var = correct_var
        self._curr_var = init_var
        self._ready_for_converge = (correct_var >= init_var)
        self._init_var = init_var
        if self._ready_for_converge:
            self._m = 0
        else:
            self._m = m

    def get_params(self, it: int, x: np.ndarray):
        # if self._curr_x is None:
        #     self._curr_x = x
        #     self._curr_var = np.array([self._curr_var]*len(x))
        # else:
        #     curr_std = np.sqrt(self._curr_var) - 10*np.sum(np.abs(self._curr_x-x),axis=1)
        #     self._ready_for_converge = (curr_std < np.sqrt(self._correct_var))
        #     curr_std[curr_std < np.sqrt(self._correct_var)] = np.sqrt(self._correct_var)
        #     self._curr_var = (curr_std)**2
        #     self._curr_x = x
        # return {'extra_var': self._curr_var}
        if it >= self._m:
            extra_var = 0
        else:
            extra_var = ( np.sqrt(self._init_var) / self._m * (self._m - it) + np.sqrt(self._correct_var) / self._m * (it) )**2 - self._correct_var
        return {'extra_var': extra_var}
    
    def ready_for_convergence(self, it: int, x: np.ndarray):
        # if self._curr_x is None:
        #     self._curr_x = x
        #     self._curr_var = np.array([self._curr_var]*len(x))
        #     self._ready_for_converge = np.array([self._correct_var >= self._init_var]*len(x))
        # else:
        #     curr_std = np.sqrt(self._curr_var) - 10*np.sum(np.abs(self._curr_x-x),axis=1)
        #     self._ready_for_converge = (curr_std < np.sqrt(self._correct_var))
        #     curr_std[curr_std < np.sqrt(self._correct_var)] = np.sqrt(self._correct_var)
        #     self._curr_var = (curr_std)**2
        #     self._curr_x = x
        # return self._ready_for_converge
        return it >= self._m

    def reset(self):
        self.curr_x = None
        self._curr_var = self._init_var
        self._ready_for_converge = (self._correct_var >= self._init_var)

class IFShortestPath(IFLikelihood, ObjectiveFunction):
    def __init__(self, meas_prop: FMCWMeasurementProperties, K=2, eps=1e-20):
        self.meas_prop = copy.deepcopy(meas_prop)
        self.if_generator = get_if_generator(meas_prop)
        self.linewidth = self.meas_prop.get_linewidth()
        self.sample_rate = self.meas_prop.get_sample_rate()
        self.if_var = self.linewidth*2/self.sample_rate/(2*np.pi)
        self.K = K
        self.snr_adjust_polyfit_coefs = np.load(os.path.join(this_dir,'phase_noise','project_normal_diff_var_polyfit.npy'))
        self.snr_adjust_polyfit_func = np.poly1d(self.snr_adjust_polyfit_coefs)
        self.eps = eps
        self.t = None
        self.ifx = None
    
    def store_observations(self, t: np.ndarray, ifx: np.ndarray):

        assert (np.ndim(t) == 1) and (np.ndim(ifx) == 1)

        if self.meas_prop.is_zero_velocity() and ( (t[-1] + 3/self.sample_rate - t[0])<(self.meas_prop.Tchirp) ):
            raise RuntimeError('not enough observations')
        elif not self.meas_prop.is_zero_velocity() and ( (t[-1] + 3/self.sample_rate - t[0])<(2*self.meas_prop.Tchirp) ):
            raise RuntimeError('not enough observations')
        
        self.t = copy.deepcopy(t)
        self.ifx = copy.deepcopy(ifx)

    def evaluate(self, x: np.ndarray, extra_var: float|None=None):

        if (np.ndim(x) != 2) or (x.shape[-1] != 2):
            raise ValueError("Dimension of x not acceptable")

        if extra_var is None:
            extra_var = 0

        if self.meas_prop.is_zero_velocity():
            if_hat = self.if_generator.get_mixed_if(self.t, x[:,0], 0, compute_deriv=False, normalize_freq=True, aliased=True)
        else:
            if_hat = self.if_generator.get_mixed_if(self.t, x[:,0], x[:,1], compute_deriv=False, normalize_freq=True, aliased=True)

        shortest_path_length = np.sum((np.mod(self.ifx-if_hat+0.5,1)-0.5)**2, axis=1)
        # nll = np.sum( -np.log(Z+self.eps), axis=1) * 2*(self.if_var+extra_var)
        return shortest_path_length
    
    def compute_gradient(self, x: np.ndarray):

        if (np.ndim(x) != 2) or (x.shape[-1] != 2):
            raise ValueError("Dimension of x not acceptable")

        if self.meas_prop.is_zero_velocity():
            if_hat, d_deriv, v_deriv = self.if_generator.get_mixed_if(self.t, x[:,0], 0, compute_deriv=True, normalize_freq=True, aliased=True)
        else:
            if_hat, d_deriv, v_deriv =self.if_generator.get_mixed_if(self.t, x[:,0], x[:,1], compute_deriv=True, normalize_freq=True, aliased=True)

        tmp = 2*(np.mod(if_hat-self.ifx+0.5,1)-0.5)
        d_deriv = np.sum(tmp * d_deriv, axis=1)
        v_deriv = np.sum(tmp * v_deriv, axis=1)
        if self.meas_prop.is_zero_velocity():
            gradient = np.stack([d_deriv, np.zeros_like(v_deriv)], axis=1)
        else:
            gradient = np.stack([d_deriv, v_deriv], axis=1)
        return gradient

    def evaluate_and_compute_gradient(self, x: np.ndarray, extra_var: float|None=None):

        if (np.ndim(x) != 2) or (x.shape[-1] != 2):
            raise ValueError("Dimension of x not acceptable")
        
        if extra_var is None:
            extra_var = 0

        if self.meas_prop.assume_zero_velocity:
            if_hat, d_deriv, v_deriv = self.if_generator.get_mixed_if(self.t, x[:,0], 0, compute_deriv=True, normalize_freq=True, aliased=True)
        else:
            if_hat, d_deriv, v_deriv = self.if_generator.get_mixed_if(self.t, x[:,0], x[:,1], compute_deriv=True, normalize_freq=True, aliased=True)
        
        shortest_path_length = np.sum((np.mod(self.ifx-if_hat+0.5,1)-0.5)**2, axis=1)
        tmp = 2*(np.mod(if_hat-self.ifx+0.5,1)-0.5)
        d_deriv = np.sum(tmp * d_deriv, axis=1)
        v_deriv = np.sum(tmp * v_deriv, axis=1)
        if self.meas_prop.is_zero_velocity():
            gradient = np.stack([d_deriv, np.zeros_like(v_deriv)], axis=1)
        else:
            gradient = np.stack([d_deriv, v_deriv], axis=1)
        return shortest_path_length, gradient

# def fmcw_restore_pixel(meas: FMCWMeasurement, n_iter=500, use_smart_grid_search=True):
#     meas_prop = meas.get_properties()
#     assume_zero_velocity = meas_prop.assume_zero_velocity
#     tf_noise_model = TimeFreqMirroredNoiseModel(meas_prop)
#     grid_generator = GridGenerator(meas_prop)

#     t, signal = meas.get_meas()
#     tf_t, tf_f = extract_inst_freq(t, signal)
#     if use_smart_grid_search:
#         grid = grid_generator.generate_smart_grid(t, signal, perturb=False)
#     else:
#         grid = grid_generator.generate_uniform_grid(perturb=False)
#     d_hat, v_hat = tf_noise_model.nll_grid_search(tf_t, tf_f, grid, return_grid_val=False)

#     for it in range(n_iter):
#         d_gradient, v_gradient = tf_noise_model.compute_nll_graident(tf_t, tf_f, d_hat, v_hat)
#         d_hat = d_hat - 1e-3 * d_gradient
#         if not assume_zero_velocity:
#             v_hat = v_hat - 1e-3 * v_gradient

#     if assume_zero_velocity:
#         return d_hat
#     return d_hat, v_hat

# def fmcw_restore2d(meas: FMCWMeasurement2d, lambd_d: float, lambd_v: float, n_iter=500):
#     height, width, num_ts = meas.get_size()
#     d_hat = np.zeros((height, width))
#     v_hat = np.zeros((height, width))
#     meas_prop = meas.get_properties()
#     tf_noise_model = TimeFreqMirroredNoiseModel(meas_prop)
#     max_d = meas_prop.get_max_d()
#     d_range = np.linspace(0, max_d, 50)
#     max_v = meas_prop.get_max_v()
#     v_range = np.linspace(-max_v, max_v, 50)
#     for i in range(height):
#         for j in range(width):
#             t, y = meas[i,j]
#             # tt = 0.5 * (t[1:] + t[:-1])
#             tt, z = extract_inst_freq(t, z)
#             d_hat[i,j], v_hat[i,j] = tf_noise_model.nll_grid_search(tt, z, d_range, v_range, perturb=False)
#     for it in range(n_iter):
#         d_gradient = np.zeros((height, width))
#         v_gradient = np.zeros((height, width))
#         for i in range(height):
#             for j in range(width):
#                 d_gradient[i,j], v_gradient[i,j] = tf_noise_model.compute_nll_graident(tt, z, d_hat[i,j], v_hat[i,j])
#         d_hat = denoise_tv_chambolle(d_hat - 1e-3 * d_gradient, weight=lambd_d)
#         v_hat = denoise_tv_chambolle(v_hat - 1e-3 * v_gradient, weight=lambd_v)

#     return d_hat, v_hat