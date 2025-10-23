
import numpy as np
from scipy.optimize import minimize_scalar, bracket
from math import factorial
from scipy.fftpack import fft, ifft
from scipy.linalg import pascal
import json
import os
from typing import Sequence
from numpy.typing import ArrayLike

from . import modulation as modf
import utils
from .meas_prop import FMCWMeasurementProperties


def compute_lorentz(freq, f0, del_freq, f_sample):
    return (del_freq/f_sample**2) * 2*np.pi / (2*np.pi**2 * ( (freq-f0)**2 + (del_freq/f_sample)**2 ) )

def gaussian_weighted_score(tf_plot, sigma, mean, freq=None):
    n_row, n_col = tf_plot.shape
    score_col = np.zeros((n_col))
    if freq is None:
        x = np.linspace(0, 1, n_row)
    else:
        x = freq
    for i_col in range(n_col):
        weights = np.exp(- (x-mean[i_col])**2 / sigma**2 / 2 )
        weights = weights / np.sum(weights)
        score_col[i_col] = np.sum(tf_plot[:,i_col] * weights)
    score = np.sum(score_col)
    return score


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

def import_meas_prop_from_config(path: str, configname: str) -> FMCWMeasurementProperties:

    f = open(path)
    config_sets = json.load(f)
    config = config_sets[configname]
    modulation = config["modulation"]
    T = config["T"]
    B = config["B"]
    sample_rate = config["sample_rate"]
    linewidth = config["linewidth"]
    carrier_wavelength = config["lambd_c"]
    reflectance = config["reflectance"]
    detector_effectivity = config["detector_effectivity"] # A/W
    transmitted_power = config["transmitted_power"] # W
    detector_effective_area = config["detector_effective_area"] # m^2
    complex_available = config["complex_available"]
    include_shot_noise = config["include_shot_noise"]
    f.close()

    meas_prop = FMCWMeasurementProperties(sample_rate, T, B, linewidth, carrier_wavelength, assume_zero_velocity=True, modulation=modulation,
                                          include_shot_noise=include_shot_noise, transmitted_power=transmitted_power, reflectance=reflectance,
                                          detector_effectivity=detector_effectivity, detector_effective_area=detector_effective_area,
                                          complex_available=complex_available)
    
    return meas_prop

def import_from_meas_data(path: str):
    meas_data = np.load(path)
    distance_arr = meas_data['distance']
    if 'velocity' in meas_data.keys():
        velocity_arr = meas_data['velocity']
        velocity_arr_exists = True
    else:
        velocity_arr_exists = False
    modulation = meas_data['modulation']
    T = meas_data['Tchirp']
    B = meas_data['bandwidth']
    sample_rate = meas_data['sample_rate']
    linewidth = meas_data['linewidth']
    carrier_wavelength = meas_data['lambd_c']
    reflectance = meas_data['reflectance']
    detector_effectivity = meas_data['detector_effectivity'] # A/W
    transmitted_power = meas_data['transmitted_power'] # W
    detector_effective_area = meas_data['detector_effective_area'] # m^2
    complex_available = meas_data['complex_available']
    include_shot_noise = meas_data['include_shot_noise']
    measurement_arr = meas_data['measurement']
    t = np.arange(0, measurement_arr.shape[3], 1) / sample_rate
    n_simulation = measurement_arr.shape[1]
    if velocity_arr_exists:
        assume_zero_velocity = False
    else:
        assume_zero_velocity = True
    # create fmcw measurement system
    meas_prop = FMCWMeasurementProperties(sample_rate, T, B, linewidth, carrier_wavelength, assume_zero_velocity=assume_zero_velocity, 
                                          modulation=modulation, include_shot_noise=include_shot_noise, transmitted_power=transmitted_power, 
                                          reflectance=reflectance, detector_effectivity=detector_effectivity, 
                                          detector_effective_area=detector_effective_area, complex_available=complex_available)
    if velocity_arr_exists:
        return_vals = (n_simulation, distance_arr, velocity_arr, measurement_arr, meas_prop)
    else:
        return_vals = (n_simulation, distance_arr, measurement_arr, meas_prop)
    return return_vals

def import_sim_data_from_meas_data(path: str):
    meas_data = np.load(path)
    distance_arr = meas_data['distance']
    if 'velocity' in meas_data.keys():
        velocity_arr = meas_data['velocity']
        velocity_arr_exists = True
    else:
        velocity_arr_exists = False
    modulation = str(meas_data['modulation'])
    T = float(meas_data['Tchirp'])
    B = float(meas_data['bandwidth'])
    sample_rate = float(meas_data['sample_rate'])
    linewidth = float(meas_data['linewidth'])
    carrier_wavelength = float(meas_data['lambd_c'])
    reflectance = float(meas_data['reflectance'])
    detector_effectivity = float(meas_data['detector_effectivity']) # A/W
    transmitted_power = float(meas_data['transmitted_power']) # W
    detector_effective_area = float(meas_data['detector_effective_area']) # m^2
    complex_available = bool(meas_data['complex_available'])
    include_shot_noise = meas_data['include_shot_noise']
    measseeds_arr = meas_data['measseeds']
    n_cycle = int(meas_data['n_cycle'])
    n_simulation = int(meas_data['n_simulation'])
    t = np.arange(0, 2*T*n_cycle, 1) / sample_rate
    if velocity_arr_exists:
        assume_zero_velocity = False
    else:
        assume_zero_velocity = True
    # create fmcw measurement system
    meas_prop = FMCWMeasurementProperties(sample_rate, T, B, linewidth, carrier_wavelength, assume_zero_velocity=assume_zero_velocity, 
                                          modulation=modulation, include_shot_noise=include_shot_noise, transmitted_power=transmitted_power, 
                                          reflectance=reflectance, detector_effectivity=detector_effectivity, 
                                          detector_effective_area=detector_effective_area, complex_available=complex_available)
    sim_data = dict()
    sim_data['n_simulation'] = int(n_simulation)
    sim_data['distance'] = distance_arr
    if velocity_arr_exists:
        sim_data['velocity'] = velocity_arr
    sim_data['measseeds'] = measseeds_arr
    sim_data['meas_prop'] = meas_prop
    sim_data['n_cycle'] = int(n_cycle)

    return sim_data

# highest order first
def compute_polynomial_derivative(t, coefs):
    x = np.zeros_like(t)
    for i, c in enumerate(coefs[-2::-1]):
        x = x + (i+1)*(t**(i))*c
    return x