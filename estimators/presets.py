
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(this_dir, '..'))
# sys.path.append(os.path.join(this_dir, '..', '..'))

import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft, ifft
from typing import Sequence
import copy

from fmcw_sys import FMCWMeasurementProperties
import optimizer
from .ifreg import IFRegressor
from .lorentzian import ConstantFrequencyEstimator, LorentzianRegressor, OracleLorentzianRegressor
from .maxpd import MaximumPeriodogram, OracleMaximumPeriodogram
from .matchfilt import MathedFilterDelayEstimator, MathedFilterDelayEstimator2D
from .freqavg import FrequencyAveraging
from .estimators_base import Estimator, OracleEstimator

def select_presets(presets: Sequence[str]):

    algo_kwargs = list()

    for i, preset in enumerate(presets):

        if preset == 'nosnradjust_uniformgrid':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'optimal', 'method':'gradient_descent',
                                              'ignore_quadrature':False, 'snr_adjustment':False, 'gd_max_n_iter':500}))

        elif preset == 'wn_snradjust_lattice':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'optimal', 'method':'gradient_descent',
                                              'init_step': 'shortestpath_likelihood',
                                              'ignore_quadrature':False, 'snr_adjustment':True, 'gd_max_n_iter':500}))
        
        elif preset == 'wn_snradjust_lattice_faster':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'optimal', 'method':'LBFGS',
                                              'init_step': 'none',
                                              'ignore_quadrature':False, 'snr_adjustment':True}))
            
        elif preset == 'wn_snradjust_lattice_avg':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'optimal', 'method':'gradient_descent',
                                              'init_step': 'shortestpath_likelihood',
                                              'ignore_quadrature':False, 'snr_adjustment':True, 'gd_max_n_iter':500, 
                                              'average':True}))
        
        
        elif preset.startswith('wn_snradjust_lattice_faster_'):
            
            str_parsed = preset.split("_")
            d_grid_num = int(str_parsed[-2])
            v_grid_num = int(str_parsed[-1])

            algo_kwargs.append((IFRegressor, {'gridgen_type':'uniform', 'method':'LBFGS',
                                              'init_step': 'none',
                                              'ignore_quadrature':False, 'snr_adjustment':True,
                                              'grid_d_num': d_grid_num, 'grid_v_num': v_grid_num}))
                             
        elif preset.startswith('wn_snradjust_lattice_'):
            
            str_parsed = preset.split("_")
            d_grid_num = int(str_parsed[-2])
            v_grid_num = int(str_parsed[-1])

            algo_kwargs.append((IFRegressor, {'gridgen_type':'uniform', 'method':'gradient_descent',
                                              'init_step': 'shortestpath_likelihood',
                                              'ignore_quadrature':False, 'snr_adjustment':True, 'gd_max_n_iter':500,
                                              'grid_d_num': d_grid_num, 'grid_v_num': v_grid_num}))
                        
        elif preset == 'shortestpath_lattice':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'optimal', 'method':'gradient_descent','likelihood':'shortest_path',
                                              'ignore_quadrature':False, 'snr_adjustment':True, 'gd_max_n_iter':500}))

        elif preset == 'nosnradjust_smartgrid':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'smart_maxpd', 'method':'gradient_descent',
                                              'ignore_quadrature':False, 'snr_adjustment':False, 'gd_max_n_iter':500}))

        elif preset == 'snradjust_smartgrid':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'smart_maxpd', 'method':'gradient_descent',
                                              'ignore_quadrature':False, 'snr_adjustment':True, 'gd_max_n_iter':500}))

        elif preset == 'nosnradjust_uniformgrid_real':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'optimal', 'method':'gradient_descent',
                                              'ignore_quadrature':True, 'snr_adjustment':False, 'gd_max_n_iter':500}))

        elif preset == 'snradjust_uniformgrid_real':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'optimal', 'method':'gradient_descent',
                                              'ignore_quadrature':True, 'snr_adjustment':True, 'gd_max_n_iter':500}))

        elif preset == 'nosnradjust_smartgrid_real':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'smart_maxpd', 'method':'gradient_descent',
                                              'ignore_quadrature':True, 'snr_adjustment':False, 'gd_max_n_iter':500}))

        elif preset == 'snradjust_smartgrid_real':

            algo_kwargs.append((IFRegressor, {'gridgen_type':'smart_maxpd', 'method':'gradient_descent',
                                              'ignore_quadrature':True, 'snr_adjustment':True, 'gd_max_n_iter':500}))

        elif preset == 'lorentz':

            algo_kwargs.append((LorentzianRegressor, {'ignore_quadrature':False}))
            
        elif preset == 'lorentz_faster':

            algo_kwargs.append((LorentzianRegressor, {'ignore_quadrature':False, 'method':'LBFGS'}))
        
        elif preset == 'maxpd':

            algo_kwargs.append((MaximumPeriodogram, {'ignore_quadrature':False}))

        elif preset == 'freqavg':

            algo_kwargs.append((FrequencyAveraging, {'ignore_quadrature':False}))

        elif preset == 'oracle_lorentz':

            algo_kwargs.append((OracleLorentzianRegressor, {'ignore_quadrature':False, 'method':'lorentzian'}))

        elif preset == 'oracle_maxpd':

            algo_kwargs.append((OracleMaximumPeriodogram, {'ignore_quadrature':False, 'method':'maxpd'}))

        elif preset == 'matchedfilter':

            algo_kwargs.append((MathedFilterDelayEstimator, {'ignore_quadrature':False}))
            
        elif preset == 'matchedfilter_optim':

            algo_kwargs.append((MathedFilterDelayEstimator, {'ignore_quadrature':False, 'optimize':True}))
            
        elif preset == 'matchedfilter2d':

            algo_kwargs.append((MathedFilterDelayEstimator2D, {'ignore_quadrature':False}))
            
        elif preset == 'matchedfilter2d_optim':

            algo_kwargs.append((MathedFilterDelayEstimator2D, {'ignore_quadrature':False, 'optimize':True}))

        elif preset == 'lorentzian_gridsearch':

            algo_kwargs.append((IFRegressor, {'gridgen_type': 'smart_lorentzian', 'method':'none', 'ignore_quadrature':False}))

        else:

            raise ValueError('unrecognized preset name')
            
    return algo_kwargs

    
def get_estimators(meas_prop: FMCWMeasurementProperties, presets: Sequence[str]) -> list[Estimator|OracleEstimator]:

    algo_kwargs = select_presets(presets)
    algos = list()
    for algo_constructor, kwargs in algo_kwargs:
        algos.append(algo_constructor(meas_prop, **kwargs))

    return algos