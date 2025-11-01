
import numpy as np
from scipy.optimize import minimize_scalar, bracket
from math import factorial
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert
from scipy.linalg import pascal
import json
import os
from typing import Sequence
from numpy.typing import ArrayLike
import time_freq

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
PARAMS_PATH = os.path.join(PROJECT_DIR, 'fmcw_sys', 'params.json')
MACHINE_EPSILON = np.finfo(np.double).eps

def estimate_if(t, y, method='polar_discriminator', mirror=False):
    if method == 'polar_discriminator':
        if y.dtype != np.complex128:
            y = hilbert(y)
        z = np.angle(y[1:] * np.conj(y[:-1])) / (2*np.pi)
        if mirror:
            z = mirror(z, 0, 0.5)
        tt = (t[1:] + t[:-1]) * 0.5
        return tt, z
    elif method == 'reassignment':
        window_length = 17
        hop_length = window_length // 4
        ht = np.arange(-window_length//2, window_length-window_length//2)
        h = np.cos(np.pi*ht/window_length)**2
        dh = 2*np.cos(np.pi*ht/window_length)*(-np.sin(np.pi*ht/window_length))*np.pi/window_length
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

def empirical_autocorr(signal):
    if signal.ndim == 2:
        N = signal.shape[1]
        M = signal.shape[0]
        normalizer = np.concatenate([np.arange(N,0,-1), np.arange(1,N)]) * M
        autocorr = np.zeros((2*N-1))
        for m in range(M):
            autocorr = autocorr + np.real(ifft(np.abs(fft(np.pad(signal[m],(0,N-1),'constant',constant_values=0)))**2))
        autocorr = np.roll(autocorr / normalizer , N)
        u = np.concatenate([-np.arange(N-1,0,-1), np.arange(0,N)])
        return u, autocorr
    else:
        N = len(signal)
        normalizer = np.concatenate([np.arange(N,0,-1), np.arange(1,N)])
        autocorr = np.real(ifft(np.abs(fft(np.pad(signal-np.mean(signal),(0,N-1),'constant',constant_values=0)))**2)) / normalizer
        autocorr = np.roll(autocorr, N)
        u = np.concatenate([-np.arange(N-1,0,-1), np.arange(0,N)])
        return u, autocorr
    
def empirical_corr(signal1, signal2):

    N = len(signal1)
    normalizer = np.concatenate([np.arange(N,0,-1), np.arange(1,N)])
    autocorr = np.real(ifft(( fft(np.pad(signal1-np.mean(signal1),(0,N-1),'constant',constant_values=0)) * np.conj(fft(np.pad(signal2-np.mean(signal2),(0,N-1),'constant',constant_values=0))) ))) / normalizer
    autocorr = np.roll(autocorr, N)
    u = np.concatenate([-np.arange(N-1,0,-1), np.arange(0,N)])
    return u, autocorr

def unfold(val, val_max, n):

    for i in range(n):
        val = 2*val_max*(i+1) - val
    return val

def get_unfold_candidate(val, val_max, n):

    if val > val_max or val < 0:
        raise ValueError('val must be in [0, val_max]')

    unfold_candidate = np.zeros((n))
    unfold_candidate[0] = val

    for i in range(1, n):
        unfold_candidate[i] = (2*i)*val_max - unfold_candidate[i-1]

    return unfold_candidate

def fold(x, min, max):
    if np.isscalar(x):
        x = np.array(x)

    less = (x < min)
    more = (x > max)

    while (np.sum(less) > 0) or (np.sum(more) > 0):
        x[less] = 2*min - x[less]
        x[more] = 2*max - x[more]
        less = (x < min)
        more = (x > max)

    return x

def mirror(x: float|np.ndarray, min: float, max: float):
    if np.isscalar(x):
        while (x < min) | (x > max):
            if x < min:
                x = 2*min - x
            else:
                x = 2*max - x

    else:    
        x = np.copy(x)

        less = (x < min)
        more = (x > max)

        while (np.sum(less) > 0) or (np.sum(more) > 0):
            x[less] = 2*min - x[less]
            x[more] = 2*max - x[more]
            less = (x < min)
            more = (x > max)

    return x

def mirror_deriv(x: float|np.ndarray, min: float, max: float):
    if np.isscalar(x):
        x = np.array(x)
    else:
        x = np.copy(x)
    deriv = np.ones(x.shape)

    less = (x < min)
    more = (x > max)

    while (np.sum(less) > 0) or (np.sum(more) > 0):
        x[less] = 2*min - x[less]
        x[more] = 2*max - x[more]
        deriv[less+more] *= -1
        less = (x < min)
        more = (x > max)

    return x, deriv

def wrap(x, min, max):
    return np.mod(x-min, max-min) + min

def unmirror(x: float|np.ndarray, val_min: float, val_max: float, n: None|int=None, n_min: None|int=None, n_max: None|int=None):
    if (n is None) and (n_min is None) and (n_max is None):
        raise ValueError("all n cannot be None")
    if (n_min is not None) and (n_max is None):
        raise ValueError()
    if (n_max is not None) and (n_min is None):
        raise ValueError()
    if (n_max is not None) and (n_min is not None) and (n_min >= n_max):
        raise ValueError()
    
    if isinstance(x, float|np.ndarray) and (n is not None):
        if isinstance(n, int):
            z = x - val_min
            n_it = np.abs(n)
            if np.sign(n) == -1:
                for i in range(np.abs(n)-1):
                    z = 2*(i+1)*(val_max-val_min) - z
                z = z * (-1)
            elif np.sign(n) == 1:
                for i in range(np.abs(n)):
                    z = 2*(i+1)*(val_max-val_min) - z
            z = z + val_min 
        elif isinstance(n, np.ndarray):
            assert x.shape == n.shape
            z = x - val_min
            k = np.copy(n)
            neg_k = np.sign(k) == -1
            pos_k = np.sign(k) == 1
            count = 0
            while np.any(pos_k):
                z[pos_k] = 2*(count+1)*(val_max-val_min) - z[pos_k]
                count = count + 1
                k[pos_k] = k[pos_k] -1
                pos_k = k > 0
            count = 0
            while np.any(neg_k):
                z[neg_k] = 2*(count)*(val_min-val_max) - z[neg_k]
                count = count + 1
                k[neg_k] = k[neg_k] + 1
                neg_k = k < 0
            z = z + val_min
        
    elif isinstance(x, float) and (n_min is not None) and (n_max is not None):
        z = np.zeros((n_max-n_min+1))
        t = x - val_min
        zero_cross = np.sign(n_min) != np.sign(n_max)
        nn_max = np.maximum(np.abs(n_min), np.abs(n_max))
        nn_min = 0 if zero_cross else np.minimum(np.abs(n_min), np.abs(n_max))
        if zero_cross:
            z[-n_min] = x
            z[-n_min-1] = 2*val_min-x
        for i in range(nn_max):
            t = 2*(i+1)*(val_max-val_min) - t
            idx1 = i + 1 - n_min
            idx2 = - i - 2 - n_min
            if (idx1 < len(z)) and (idx1 >= 0):
                z[idx1] = t + val_min
            if (idx2 < len(z)) and (idx2 >= 0):
                z[idx2] = t*(-1) + val_min 
    return z

def _get_polynomial(t, coefs):
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    x = np.zeros_like(t)
    for i, c in enumerate(coefs):
        x = x + t**(i+1)*c
    return x

def hilbert_transform(signal):
    assert signal.dtype == float
    Fx = fft(signal)
    idx = np.arange(len(Fx))
    N = len(Fx)
    pos_freq = ( idx < (N-N//2) ) * ( idx > 0)
    neg_freq = ( idx >= (N-N//2) )
    Fx[pos_freq] = Fx[pos_freq] * -1j
    Fx[neg_freq] = Fx[neg_freq] * 1j
    Fx[0] = 0
    return signal + 1j* np.real(ifft(Fx))

# highest order first
def compute_polynomial_derivative(t, coefs):
    x = np.zeros_like(t)
    for i, c in enumerate(coefs[-2::-1]):
        x = x + (i+1)*(t**(i))*c
    return x

def compute_polynomial_integral(coefs, x0, x1):
    x = 0
    for i, c in enumerate(coefs[-1::-1]):
        x = x + (x1**(i+1))*c/(i+1) - (x0**(i+1))*c/(i+1)
    return x