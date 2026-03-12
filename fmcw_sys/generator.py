
import numpy as np
from scipy.optimize import minimize_scalar, bracket
from math import factorial
from scipy.fftpack import fft, ifft
from scipy.linalg import pascal
import json
import os
from typing import Sequence
from numpy.typing import ArrayLike
import time

from .meas_prop import FMCWMeasurementProperties
from . import modulation as modf
import utils

def gen_phase_noise(ts: ArrayLike, 
                    distance: float|ArrayLike, 
                    velocity: float|ArrayLike, 
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


class FMCWMeasurement():
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        self.meas_prop = meas_prop
    
    def assign(self, t, meas):
        self.meas_time = t
        self.meas_val = meas

    def generate_phase(self, ts: np.ndarray, distance: float, velocity: float):

        T_chirp = self.meas_prop.get_chirp_length()
        bandwidth = self.meas_prop.get_bandwidth()
        mode = self.meas_prop.get_modulation_type()
        center_wavelength = self.meas_prop.get_carrier_wavelength()
        mixed_coef = self.meas_prop.get_mix_coef()
        poly_coefs_up, poly_coefs_down = self.meas_prop.get_poly_coefs()

        delay = distance * 2 / 3e8
        gamma = bandwidth / T_chirp
        v = 2*velocity/3e8
        f0 = 3e8 / center_wavelength - bandwidth / 2

        if mode == 'sawtooth':

            ts_w = np.mod(ts-delay-v*ts, T_chirp)
            phase = 0.5*gamma*(ts_w**2)
            period_num = np.floor((ts-delay-v*ts)/T_chirp).astype(int)
            phase_offset = period_num * 0.5*gamma*(T_chirp**2)
            phase_rx = phase + phase_offset - f0*v*ts

        elif mode == 'triangle' or mode == 'triangle_phase':

            ts_w = utils.mirror(ts-delay-v*ts, 0, T_chirp)
            phase = 0.5*gamma*(ts_w**2)
            period_num = np.floor((ts-delay-v*ts)/T_chirp).astype(int)
            phase = phase* (-np.ones(ts.shape))**(period_num)
            phase_offset1 = ( (period_num % 2) == 1 ) * 0.5*gamma*(T_chirp**2)
            phase_offset2 = period_num * 0.5*gamma*(T_chirp**2)
            phase_rx = phase + phase_offset1 + phase_offset2 - f0*v*ts

        elif mode == 'sinusoidal':
            phase_rx = -1/np.pi/2*T_chirp*bandwidth*np.sin(np.pi*(ts-delay-v*ts)/T_chirp) + bandwidth/2*(ts-delay-v*ts) - f0*v*ts

        elif mode == 'mixed':

            ts_w = utils.mirror(ts-delay-v*ts, 0, T_chirp)
            phase = 0.5*gamma*(ts_w**2)
            period_num = np.floor((ts-delay-v*ts)/T_chirp).astype(int)
            phase = phase* (-np.ones(ts.shape))**(period_num)
            phase_offset1 = ( (period_num % 2) == 1 ) * 0.5*gamma*(T_chirp**2)
            phase_offset2 = period_num * 0.5*gamma*(T_chirp**2)
            phase_rx = (phase + phase_offset1 + phase_offset2) * mixed_coef + (-1/np.pi/2*T_chirp*bandwidth*np.sin(np.pi*(ts-delay-v*ts)/T_chirp) + bandwidth/2*(ts-delay-v*ts)) * (1-mixed_coef) - f0*v*ts

        elif mode =='polynomial':
            if poly_coefs_up is None or poly_coefs_down is None:
                raise ValueError('polynomial coefficients must be specified when selecting pps model')
            
            up_total = utils._get_polynomial(T_chirp, poly_coefs_up)
            down_total = utils._get_polynomial(T_chirp, poly_coefs_down)

            ts_rx = ts-delay-v*ts
            Tstart = np.floor(ts_rx[0]/T_chirp/2)*T_chirp*2
            Tend = np.ceil(ts_rx[-1]/T_chirp/2)*T_chirp*2
            phase_rx = np.zeros((len(ts)))
            offset = 0
            for T in np.arange(Tstart, Tend, 2*T_chirp):
                interval1 = (T <= ts_rx)*(ts_rx < T+T_chirp)
                interval2 = (T+T_chirp <= ts_rx)*(ts_rx <= T+2*T_chirp)
                phase_rx[interval1] = offset + utils._get_polynomial(ts_rx[interval1]-T, poly_coefs_up)
                phase_rx[interval2] = offset + up_total + utils._get_polynomial(ts_rx[interval2]-T-T_chirp, poly_coefs_down)
                offset = offset + up_total + down_total
            phase_rx = phase_rx - f0*v*ts

        elif mode == 'smoothstairs':

            phase = modf.SmoothStairsModulation(self.meas_prop).generate_phase_x(ts, distance, velocity)
            phase_rx = phase / (2*np.pi)
        
        elif mode == 'hardsteep':

            phase = modf.HardSteepModulation(self.meas_prop).generate_phase_x(ts, distance, velocity)
            phase_rx = phase / (2*np.pi)


        return 2*np.pi*phase_rx

    def generate_reference_phase(self, ts: np.ndarray, distance: float, velocity: float):
        print("generate_reference_phase is deprecated")
        return self.generate_phase(ts, distance, velocity)

    def generate(self, distance: float|ArrayLike, velocity: float|ArrayLike, ts: np.ndarray, 
                 include_phase_noise: bool|None=None, include_shot_noise : bool|None=None,
                 distance_fall_off: bool=True,
                 random_phase_offset: bool=True,
                 return_complex_signal :bool|None=None):
        
        if np.isscalar(distance) and np.isscalar(velocity):
            n_echo = 1
            distance = np.array([distance])
            velocity = np.array([velocity])
        else:
            if len(distance) != len(velocity):
                raise ValueError('distance and velocity must have same length')
            n_echo = len(distance)
            distance = np.array(distance)
            velocity = np.array(velocity)

        T_chirp = self.meas_prop.get_chirp_length()
        bandwidth = self.meas_prop.get_bandwidth()
        mode = self.meas_prop.get_modulation_type()
        linewidth = self.meas_prop.get_linewidth()
        center_wavelength = self.meas_prop.get_carrier_wavelength()
        mixed_coef = self.meas_prop.get_mix_coef()
        poly_coefs_up, poly_coefs_down = self.meas_prop.get_poly_coefs()
        if include_phase_noise is None:
            include_phase_noise = True
        if include_shot_noise is None:
            include_shot_noise = self.meas_prop.include_shot_noise
        if return_complex_signal is None:
            return_complex_signal = self.meas_prop.is_complex_available()

        delay = distance * 2 / 3e8
        gamma = bandwidth / T_chirp
        v = 2*velocity/3e8
        f0 = 3e8 / center_wavelength - bandwidth / 2

        phase_tx = self.generate_phase(ts, 0, 0)
        phase_rx = np.zeros((n_echo, len(ts)))
        for i, (d, v) in enumerate(zip(distance, velocity)):
            phase_rx[i] = self.generate_phase(ts, d, v)
        mixed_phase = phase_tx - phase_rx 
        if random_phase_offset:
            pass
        if include_phase_noise:
            phase_noise = gen_phase_noise(ts, distance, velocity, linewidth)
            mixed_phase = mixed_phase + phase_noise
        
        R = self.meas_prop.reflectance
        A = self.meas_prop.detector_effective_area
        eta = self.meas_prop.detector_effectivity
        tx_power = self.meas_prop.transmitted_power
        lo_power = tx_power
        q = 1.6e-19
        if distance_fall_off:
            rx_power = tx_power * A / (np.pi*distance**2) * R
        else:
            rx_power = tx_power*np.ones_like(distance)
        rx_interference = 0
        for i in range(n_echo):
            rx_interference += rx_power[i]
            for j in range(0,i):
                rx_interference += 2*np.sqrt(rx_power[i]*rx_power[j])*np.cos(phase_rx[i]-phase_rx[j])

        if return_complex_signal:
            y1 = 0.5*eta*(lo_power + rx_interference)*(1+1j) + eta*np.sqrt(lo_power*rx_power) @ np.exp(1j*(mixed_phase))
            y2 = 0.5*eta*(lo_power + rx_interference)*(1+1j) - eta*np.sqrt(lo_power*rx_power) @ np.exp(1j*(mixed_phase))
        else:
            y1 = 0.5*eta*(lo_power + rx_interference) + eta*np.sqrt(lo_power*rx_power) @ np.cos(mixed_phase)
            y2 = 0.5*eta*(lo_power + rx_interference) - eta*np.sqrt(lo_power*rx_power) @ np.cos(mixed_phase)

        if include_shot_noise:
            y1 = y1 + np.sqrt(q*np.real(y1)*(np.real(y1)>0))*np.random.randn(*y1.shape)
            y1 = y1 + 1j*np.sqrt(q*np.imag(y1)*(np.imag(y1)>0))*np.random.randn(*y1.shape)
            y2 = y2 + np.sqrt(q*np.real(y2)*(np.real(y2)>0))*np.random.randn(*y2.shape)
            y2 = y2 + 1j*np.sqrt(q*np.imag(y2)*(np.imag(y2)>0))*np.random.randn(*y2.shape)
        
        u = y1 - y2
        v = y1 + y2

        return_vals = (u, v)

        return return_vals

    def get_meas(self):
        return self.meas_time, self.meas_val
    
    def get_properties(self):
        return self.meas_prop

