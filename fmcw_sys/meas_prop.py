
import numpy as np
from scipy.optimize import minimize_scalar, bracket
from math import factorial
from scipy.fftpack import fft, ifft
from scipy.linalg import pascal
import json
import os
from typing import Sequence
from numpy.typing import ArrayLike

import utils

class FMCWMeasurementProperties():
    def __init__(self, sample_rate, Tchirp, bandwidth, linewidth, lambd_c, modulation, 
                 mix_coef=0.99, poly_coefs_up=None, poly_coefs_down=None, assume_zero_velocity=False,
                 include_shot_noise=True, transmitted_power=1e-3, reflectance=0.1, detector_effectivity=1, detector_effective_area=1e-6, complex_available=True):
        self.sample_rate = sample_rate
        self.Tchirp = Tchirp
        self.bandwidth = bandwidth
        self.linewidth = linewidth
        self.lambd_c = lambd_c
        self.modulation = modulation
        self.mix_coef = mix_coef
        self.poly_coefs_up = poly_coefs_up
        self.poly_coefs_down = poly_coefs_down
        self.max_d = 2*self.Tchirp * 3e8 / 2
        self.max_v = self.sample_rate * self.lambd_c / 4
        self.transmitted_power = transmitted_power
        self.assume_zero_velocity = assume_zero_velocity
        self.reflectance = reflectance
        self.detector_effectivity = detector_effectivity
        self.detector_effective_area = detector_effective_area
        self.include_shot_noise = include_shot_noise
        self.complex_available = complex_available
        self.boundary_constraint = False
        self.d_range = (0, self.max_d)
        self.v_range = (-self.max_v, self.max_v)
        # by default lower limit is not included in the interval but upper limit is

    def compute_expected_snr(self, distance):
        eta = self.detector_effectivity
        R = self.reflectance
        A = self.detector_effective_area
        tx_power = self.transmitted_power
        lo_power = self.transmitted_power
        rx_power = tx_power * A / (np.pi*distance**2) * R
        q = 1.6e-19
        return 10*np.log10((2 * eta**2 *lo_power *rx_power)/(1*q*eta*(lo_power+rx_power)))

    def compute_expected_total_power(self, distance):
        eta = self.detector_effectivity
        R = self.reflectance
        A = self.detector_effective_area
        tx_power = self.transmitted_power
        lo_power = self.transmitted_power
        rx_power = tx_power * A / (np.pi*distance**2) * R
        q = 1.6e-19
        return (2* eta**2 *lo_power *rx_power)+(1*q*eta*(lo_power+rx_power))
    
    def compute_expected_signal_power(self, distance):
        eta = self.detector_effectivity
        R = self.reflectance
        A = self.detector_effective_area
        tx_power = self.transmitted_power
        lo_power = self.transmitted_power
        rx_power = tx_power * A / (np.pi*distance**2) * R
        q = 1.6e-19
        if self.is_complex_available():
            expected_signal_power = (2 * eta**2 *lo_power *rx_power)
        else:
            expected_signal_power = (2* eta**2 *lo_power *rx_power)
        return expected_signal_power

    def compute_expected_shotnoise_power(self, distance):
        eta = self.detector_effectivity
        R = self.reflectance
        A = self.detector_effective_area
        tx_power = self.transmitted_power
        lo_power = self.transmitted_power
        rx_power = tx_power * A / (np.pi*distance**2) * R
        q = 1.6e-19
        if self.is_complex_available():
            expected_shotnoise_power = (4*q*eta*(lo_power+rx_power))
        else:
            expected_shotnoise_power = (2*q*eta*(lo_power+rx_power))
        return expected_shotnoise_power

    def set_boundary_constraint(self, d_range: Sequence[float]|None=None, v_range: Sequence[float]|None=None):
        if d_range is not None:
            if len(d_range) != 2:
                raise ValueError('d_range must be length of 2')
            if d_range[0] < 0:
                raise ValueError('lower limit should not be less than 0')
            if d_range[1] > self.max_d:
                raise ValueError('upper limit should not be more than {:.2f}'.format(self.max_d))
            self.d_range = d_range
            
        if v_range is not None:
            if len(v_range) != 2:
                raise ValueError('v_range must be length of 2')
            if v_range[0] < -self.max_v:
                raise ValueError('lower limit should not be less than 0')
            if v_range[1] > self.max_v:
                raise ValueError('upper limit should not be more than {:.2f}'.format(self.max_d))
            self.v_range = v_range
        self.boundary_constraint = True

    def remove_boundary_constraint(self):
        self.d_range = (0, self.max_d)
        self.v_range = (-self.max_v, self.max)
        self.boundary_constraint = False
            
    def set_max_range(self, percentage):
        self.max_d = (2*self.Tchirp * percentage - 1/self.sample_rate) * 3e8 / 2
        self.full_range = False

    def get_range(self):
        return self.d_range, self.v_range

    def get_sample_rate(self):
        return self.sample_rate

    def get_chirp_length(self):
        return self.Tchirp

    def get_bandwidth(self):
        return self.bandwidth
    
    def get_linewidth(self):
        return self.linewidth
    
    def get_carrier_wavelength(self):
        return self.lambd_c
    
    def get_modulation_type(self):
        return self.modulation
    
    def get_mix_coef(self):
        return self.mix_coef
    
    def get_poly_coefs(self):
        return self.poly_coefs_up, self.poly_coefs_down
    
    def get_max_d(self):
        return self.max_d
    
    def get_max_v(self):
        return self.max_v
    
    def is_complex_available(self):
        return self.complex_available

    def is_zero_velocity(self):
        return self.assume_zero_velocity
    
    def where_is_constant_beat_frequency(self, t: np.ndarray, distance: float) -> np.ndarray[bool]:
        if self.modulation != 'triangle':
            raise RuntimeError('this method is only valid for triangular modulation')
        delay = distance * 2 / 3e8
        time_cushion = 10/self.sample_rate
        T = self.Tchirp
        if delay < T:
            cbf_region1 = ((T-time_cushion)>t)*(t>=(delay+time_cushion))
            cbf_region2 = ((2*T-time_cushion)>t)*(t>=(T+(delay+time_cushion)))
        elif (delay >=T) and (delay < 2*T):
            cbf_region2 = ((T+time_cushion)<t)*(t<=(delay-time_cushion))
            cbf_region1 = (time_cushion<t)*(t<=(delay-time_cushion-T))
        else:
            raise RuntimeError('distance cannot be resolved')
        return cbf_region1, cbf_region2
