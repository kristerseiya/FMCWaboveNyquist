
import numpy as np
import copy

from .estimators_base import Estimator
from fmcw_sys import FMCWMeasurement, FMCWMeasurementProperties
import utils

class FrequencyAveraging(Estimator):
    def __init__(self, meas_prop: FMCWMeasurementProperties, ignore_quadrature=False):

        if ignore_quadrature:
            raise RuntimeError("complex signal required for this method")
        
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
        self.fmcw_meas = FMCWMeasurement(meas_prop)
        
        
    def estimate(self, t: np.ndarray, mixed_signal: np.ndarray, second_output: np.ndarray|None=None):

        if (np.ndim(t)!=1) or (np.ndim(mixed_signal)!=1):
            raise ValueError('input dimension is incorrect')
        if self.assume_zero_velocity and ( (t[-1] + 2/self.meas_prop.sample_rate - t[0])<(self.meas_prop.Tchirp) ):
            raise RuntimeError('not enough observations')
        elif not self.assume_zero_velocity and ( (t[-1] + 2/self.meas_prop.sample_rate - t[0])<(2*self.meas_prop.Tchirp) ):
            raise RuntimeError('not enough observations')
        
        if self.test_flag:
            intermediate_results = dict()

        if self.ignore_quadrature:
            mixed_signal = np.real(mixed_signal)
            if second_output is not None:
                second_output = np.real(second_output)
        
        tt, ifx = utils.estimate_if(t, mixed_signal, method='polar_discriminator', mirror=False)
        
        if self.test_flag:
            intermediate_results['if_estimate'] = (tt, ifx)

        v_hat = np.mean(ifx) * self.lambd_c / 2 * self.sample_rate
        
        phase = self.fmcw_meas.generate_phase(t, 0, 0)
        if_tx = (phase[1:]-phase[:-1]) * self.sample_rate / (2*np.pi)
        # V = 4*np.pi*np.mean(np.abs(if_tx))/3e8 #/ self.sample_rate
        V = 2*np.pi*np.mean(np.abs(if_tx))*2/3e8 #/ self.sample_rate


        phase_hat = np.angle(mixed_signal)
        phase_hat = np.unwrap(phase_hat)
        ft = 2*np.pi*2/self.lambd_c*v_hat*t
        tmp = phase_hat - ft
        #tmp = utils.wrap(phase_hat - ft, -np.pi, np.pi)
        # tmp = utils.wrap(phase_hat, -np.pi, np.pi)
        phi_avg = np.mean( np.abs(tmp) ) #/ self.sample_rate
        d_hat = phi_avg / V #* self.sample_rate

        x_hat = np.array([d_hat, v_hat])
        
        if self.test_flag:
            intermediate_results['if_tx'] = (tt, if_tx)
            intermediate_results['phase_hat'] = phase_hat
        
        if not self.test_flag:
            return_val = x_hat
        else:
            return_val = (x_hat, intermediate_results)

        return return_val
    
    def set_test_flag(self, test: bool):
        self.test_flag = test