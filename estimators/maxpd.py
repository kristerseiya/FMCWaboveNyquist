
import numpy as np
import copy
from scipy.fft import fft
from scipy.optimize import least_squares, curve_fit, bracket, minimize_scalar

from fmcw_sys import FMCWMeasurement, FMCWMeasurementProperties
import utils
import optimizer
from .estimators_base import Estimator, OracleEstimator

def max_periodogram(t: np.ndarray, signal: np.ndarray, method='fine', meas_prop: FMCWMeasurementProperties|None=None):

    N = len(signal)
    periodogram = np.abs(fft(signal, norm='ortho'))**2
    periodogram /= np.linalg.norm(signal)**2
    periodogram = periodogram.astype(float)
    if meas_prop is not None:
        sample_rate = meas_prop.get_sample_rate()
    else:
        sample_rate = 1/(t[1] - t[0])
    t = t * sample_rate

    real = False
    if meas_prop is not None and meas_prop.complex_available == False:
            real = True
    
    if (meas_prop is None) and (not np.iscomplexobj(signal)):
        real = True
    
    if real:
        periodogram = periodogram[:(N-N//2)]
        # freq = np.fft.fftfreq(N, 1)
        freq = np.arange(N) / N
        freq = freq[:(N-N//2)]
        # linewidth = 2*np.pi*linewidth #/ (2*np.pi)**2
    else:
        # freq = np.fft.fftfreq(N, 1)
        freq = np.arange(N) / N

    max_idx = np.argmax(periodogram)
    #print(f'{len(t)}, {len(signal)}, {max_idx}/{len(freq)}',)
    if method == 'coarse':
        pdgm_max = freq[max_idx]
        success = True
    elif method == 'fine':
        # if max_idx == 0:
        #     brckt = (freq[0], freq[1])
        #     bounds = (freq[0], freq[1])
        # elif max_idx == (len(freq)-1):
        #     brckt = (freq[max_idx-1], freq[max_idx])
        #     bounds = (freq[max_idx-1], freq[max_idx])
        # else:
        #     bounds = (freq[max_idx-1], freq[max_idx+1])
        #     # scipy bracket
        #     xa, xb, xc, _, _, _, _ = bracket(lambda f: -np.abs(np.exp(1j * 2*np.pi * f * t) @ signal)**2, xa=freq[max_idx-1], xb=freq[max_idx])
        #     brckt = (xa, xb, xc)
        #     # bracket = (freqs[np.argmax(pdgm)-1], freqs[np.argmax(pdgm)], freqs[np.argmax(pdgm)+1])
        # try:
        #     opt_res = minimize_scalar(lambda f: -np.abs(np.exp(-1j * 2*np.pi * f * t) @ signal)**2, 
        #                               bracket=brckt, bounds=bounds, method='bounded')
        # # try:
        # #     opt_res = minimize_scalar(lambda f: -np.abs(np.exp(-1j * 2*np.pi * f * t) @ signal)**2, 
        # #                               bounds=bounds, method='golden')
        #     pdgm_max = opt_res.x
        #     success = True
        # except:
        #     pdgm_max = freq[max_idx]
        #     success = False
        pdgm_max = freq[max_idx]
        success = True
        signal_normalized = signal / np.sqrt(np.mean(np.square(np.abs(signal))))
        def obj_func(f):
            return -np.abs(np.dot(np.exp(-1j * 2*np.pi * f * t),signal_normalized))**2
        res = minimize_scalar(obj_func, 
                        bracket=(pdgm_max-1/N, pdgm_max, pdgm_max+1/N),
                        method="Brent")
        pdgm_max = res.x
    pdgm_max = utils.wrap(pdgm_max, -0.5, 0.5)
    return pdgm_max*sample_rate, success

class MaximumPeriodogram(Estimator):
    def __init__(self, meas_prop: FMCWMeasurementProperties, 
                 ignore_quadrature=False, remap=True):

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
        self.remap = remap
        self.test_flag = False

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
            pdgm_max1, success1 = max_periodogram(t[meas1_idx], mixed_signal[meas1_idx], 
                                                                method='fine', meas_prop=self.meas_prop)
            beat_freq1 = pdgm_max1 / self.sample_rate
            pdgm_max2, success2 = max_periodogram(t[meas2_idx], mixed_signal[meas2_idx], 
                                                            method='fine', meas_prop=self.meas_prop)
            beat_freq2 = pdgm_max2 / self.sample_rate

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
            d_hat = (freq1-freq2)/2  * self.sample_rate / gamma * 3e8 / 2
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
    
class OracleMaximumPeriodogram(OracleEstimator):
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
            d_hat = (freq1+freq2) * self.sample_rate / gamma * 3e8 / 2
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
    