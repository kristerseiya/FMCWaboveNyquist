
import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft, ifft
from typing import Sequence
import copy

import utils
from .meas_prop import FMCWMeasurementProperties

class Modulation():
    def __init__(self):
        pass
    def generate_phase_x(self, distance: float, velocity: float):
        raise NotImplementedError()
    
    def generate_freq_x(self):
        raise NotImplementedError()
    
    def generate_phase(self, distance: float, velocity: float):
        return self.generate_phase_x(0, 0) - self.generate_phase_x(distance, velocity)
    
    def generate_freq(self, distance: float, velocity: float):
        return self.generate_freq_x(0, 0) - self.generate_freq_x(distance, velocity)
    
class SinusoidalModulation(Modulation):
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        if meas_prop.get_modulation_type() != 'sinusoidal':
            raise ValueError('measurement property is not consistent with the class')
        self.meas_prop = copy.deepcopy(meas_prop)
        self.sample_rate = meas_prop.get_sample_rate()
        self.Tchirp = meas_prop.get_chirp_length()
        self.bandwidth = meas_prop.get_bandwidth()
        self.linewidth = meas_prop.get_linewidth()
        self.lambd_c = meas_prop.get_carrier_wavelength()
    
    def generate_freq(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                     compute_jacobian: bool=False, normalize_freq: bool=True, aliased: bool=True):
     
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
            mixed_if = sin_if_tx - sin_if_rx
            aliased_deriv = 1

        if compute_jacobian:
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

    def generate_freq_x(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_jacobian: bool=False, normalize_freq: bool=True):
        
        doppler_shift = velocity * 2 / self.lambd_c
        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        sin_if_rx = - 0.5 * self.bandwidth * np.cos(np.pi/self.Tchirp*(t-delay)) - doppler_shift
        sin_if_rx += 0.5 * self.bandwidth
        if normalize_freq:
            sin_if_rx /= self.sample_rate

        if compute_jacobian:
            raise NotImplementedError('compute jacobian for sinusoidal modulation generate_freq_x is not supported')

        return sin_if_rx
    
    def generate_phase_x(self, t: np.ndarray, 
                         distance: float|np.ndarray=0, velocity: float|np.ndarray=0, 
                         compute_jacobian: bool=False, compute_hessian: bool=False):
        
        if isinstance(distance, np.ndarray):
            distance = np.expand_dims(distance, 1)
        if isinstance(velocity, np.ndarray):
            velocity = np.expand_dims(velocity, 1)

        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        v = 2*velocity/3e8
        f0 = 3e8 / self.lambd_c - self.bandwidth / 2

        phase = -1/np.pi/2*self.Tchirp*self.bandwidth*np.sin(np.pi*(t-delay-v*t)/self.Tchirp) 
        phase += self.bandwidth/2*(t-delay-v*t) - f0*v*t
        phase *= 2*np.pi

        if compute_jacobian or compute_hessian:
            if isinstance(distance, np.ndarray):
                jacobian = np.zeros((len(distance), len(t), 2))
                jacobian[:,:,0] = 1/2*self.bandwidth*np.cos(np.pi*(t-delay-v*t)/self.Tchirp) 
                jacobian[:,:,0] += -self.bandwidth/2
                jacobian[:,:,0] *= 2 / 3e8
                jacobian[:,:,1] = 1/2*self.bandwidth*np.cos(np.pi*(t-delay-v*t)/self.Tchirp)*t 
                jacobian[:,:,1] += -self.bandwidth/2*t - f0*t
                jacobian[:,:,1] *= 2 / 3e8
            else:
                jacobian = np.zeros((len(t), 2))
                jacobian[:,0] = 1/2*self.bandwidth*np.cos(np.pi*(t-delay-v*t)/self.Tchirp) 
                jacobian[:,0] += -self.bandwidth/2
                jacobian[:,0] *= 2 / 3e8
                jacobian[:,1] = 1/2*self.bandwidth*np.cos(np.pi*(t-delay-v*t)/self.Tchirp)*t 
                jacobian[:,1] += -self.bandwidth/2*t - f0*t
                jacobian[:,1] *= 2 / 3e8
            jacobian *= 2*np.pi
        
        if compute_hessian:
            if isinstance(distance, np.ndarray):
                hessian = np.zeros((len(distance), len(t), 3))
                hessian[:,:,0] = 1/2*self.bandwidth/self.Tchirp*np.pi*np.sin(np.pi*(t-delay-v*t)/self.Tchirp) 
                hessian[:,:,0] *= (2 / 3e8)**2
                hessian[:,:,1] = 1/2*self.bandwidth/self.Tchirp*np.pi*np.sin(np.pi*(t-delay-v*t)/self.Tchirp)*(t **2)
                hessian[:,:,1] *= (2 / 3e8)**2
                hessian[:,:,2] = 1/2*self.bandwidth/self.Tchirp*np.pi*np.sin(np.pi*(t-delay-v*t)/self.Tchirp)*t
                hessian[:,:,2] *= (2 / 3e8)**2
            else:
                hessian = np.zeros((len(t), 3))
                hessian[:,0] = 1/2*self.bandwidth/self.Tchirp*np.pi*np.sin(np.pi*(t-delay-v*t)/self.Tchirp) 
                hessian[:,0] *= (2 / 3e8)**2
                hessian[:,1] = 1/2*self.bandwidth/self.Tchirp*np.pi*np.sin(np.pi*(t-delay-v*t)/self.Tchirp)*(t **2)
                hessian[:,1] *= (2 / 3e8)**2
                hessian[:,2] = 1/2*self.bandwidth/self.Tchirp*np.pi*np.sin(np.pi*(t-delay-v*t)/self.Tchirp)*t
                hessian[:,2] *= (2 / 3e8)**2
            hessian *= 2*np.pi
        
        if compute_hessian:
            return phase, jacobian, hessian
        elif compute_jacobian:
            return phase, jacobian
        return phase
    
    def generate_phase(self, t: np.ndarray, 
                       distance: float|np.ndarray=0, velocity: float|np.ndarray=0, 
                       compute_jacobian: bool=False, compute_hessian: bool=False):
        
        phase_tx = self.generate_phase_x(t, 0, 0)
        if isinstance(distance, np.ndarray):
            phase_tx = np.expand_dims(phase_tx, axis=0)

        if compute_hessian:
            phase_rx, jacobian_rx, hessian_rx = self.generate_phase_x(t, distance, velocity, compute_hessian=True)
            
            return phase_tx-phase_rx,  -jacobian_rx, -hessian_rx
        
        if compute_jacobian:
            phase_rx, jacobian_rx = self.generate_phase_x(t, distance, velocity, compute_jacobian=True)
            
            return phase_tx-phase_rx, -jacobian_rx
        
        phase_rx = self.generate_phase_x(t, distance, velocity)
        return phase_tx-phase_rx

class TriangularModulation(Modulation):
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        if meas_prop.get_modulation_type() != 'triangle':
            raise ValueError('measurement property is not consistent with the class')
        self.meas_prop = copy.deepcopy(meas_prop)
        self.sample_rate = meas_prop.get_sample_rate()
        self.Tchirp = meas_prop.get_chirp_length()
        self.bandwidth = meas_prop.get_bandwidth()
        self.linewidth = meas_prop.get_linewidth()
        self.lambd_c = meas_prop.get_carrier_wavelength()
    
    def generate_freq(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                     compute_jacobian: bool=False, normalize_freq: bool=True, aliased: bool=True):
        
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
            mixed_if = triangle_if_tx - triangle_if_rx
            aliased_deriv = 1

        if compute_jacobian:
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

    def generate_freq_x(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_jacobian: bool=False, normalize_freq: bool=True):
        
        sample_rate = self.meas_prop.get_sample_rate()
        doppler_shift = velocity * 2 / self.lambd_c
        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        triangle_if_tx = gamma*utils.mirror(t, 0, self.Tchirp)
        t_mirrored, mirror1_deriv = utils.mirror_deriv(t-delay, 0, self.Tchirp)
        triangle_if_rx = gamma*t_mirrored - doppler_shift

        if normalize_freq:
            triangle_if_rx  /= sample_rate

        if compute_jacobian:
            raise NotImplementedError('compute jacobian for triangular modulation generate_freq_x is not supported')

        return triangle_if_rx
    
    def generate_phase_x(self, t: np.ndarray, 
                         distance: float|np.ndarray=0, velocity: float|np.ndarray=0,
                         compute_jacobian: bool=False):

        if isinstance(distance, np.ndarray):
            distance = np.expand_dims(distance, 1)
        if isinstance(velocity, np.ndarray):
            velocity = np.expand_dims(velocity, 1)

        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        v = 2*velocity/3e8
        f0 = 3e8 / self.lambd_c - self.bandwidth / 2

        ts_w, mirror_deriv = utils.mirror_deriv(t-delay-v*t, 0, self.Tchirp)
        phase = 0.5*gamma*(ts_w**2)
        period_num = np.floor((t-delay-v*t)/self.Tchirp).astype(int)
        phase = phase* (-np.ones(t.shape))**(period_num)
        phase_offset1 = ( (period_num % 2) == 1 ) * 0.5*gamma*(self.Tchirp**2)
        phase_offset2 = period_num * 0.5*gamma*(self.Tchirp**2)
        phase = phase + phase_offset1 + phase_offset2 - f0*v*t
        phase *= 2*np.pi

        if compute_jacobian:
            if isinstance(distance, np.ndarray):
                jacobian = np.zeros((len(distance), len(t), 2))
                jacobian[:,:,0] = gamma*ts_w*mirror_deriv*(-2/3e8)
                jacobian[:,:,1] = gamma*ts_w*mirror_deriv*(-2*t/3e8)
                jacobian[:,:,1] += -f0*(2*t/3e8)
            else:
                jacobian = np.zeros((len(t), 2))
                jacobian[:,0] = gamma*ts_w*mirror_deriv*(-2/3e8)
                jacobian[:,1] = gamma*ts_w*mirror_deriv*(-2*t/3e8)
                jacobian[:,1] += -f0*(2*t/3e8)
            jacobian *= 2*np.pi
            return phase, jacobian
        
        return phase
    
    def generate_phase(self, t: np.ndarray, 
                       distance: float|np.ndarray=0, velocity: float|np.ndarray=0, 
                       compute_jacobian: bool=False):
        phase_tx = self.generate_phase_x(t, 0, 0)
        if isinstance(distance, np.ndarray):
            phase_tx = np.expand_dims(phase_tx, axis=0)

        if compute_jacobian:
            phase_rx, jacobian_rx = self.generate_phase_x(t, distance, velocity, compute_jacobian=True)
            
            return phase_tx-phase_rx, -jacobian_rx
        
        phase_rx = self.generate_phase_x(t, distance, velocity)
        return phase_tx-phase_rx


def _stairfunc(x,c,e,x0,d,a,compute_deriv=False):
    x_w = utils.mirror(x, 0, 1)
    y = np.zeros_like(x_w)
    xint1 = (x_w>=-x0)*(x_w<x0)
    xint2 = (x_w>=x0)*(x_w<(1-x0))
    xint3 = (x_w>=(1-x0))*(x_w<(1+x0))
    y[xint1] = a*(x_w[xint1])**2
    y[xint2] = (x_w[xint2]-0.5)-d*np.sin(c*(x_w[xint2]-0.5)) + e
    y[xint3] = -a*(x_w[xint3]-1)**2 + 1
    y = y - 0.5
    if compute_deriv:
        z = np.zeros_like(x)
        z[xint1] = 2*a*(x_w[xint1])
        z[xint2] = 1-c*d*np.cos(c*(x_w[xint2]-0.5))
        z[xint3] = -2*a*(x_w[xint3]-1)
        period_num = np.floor(x).astype(int)
        z = z * (-np.ones(z.shape))**(period_num)
        return y, z
    return y

def _stairfunc_integral(x,c,e,x0,d,a):
    
    x_w = utils.mirror(x, 0, 1)
    y = np.zeros_like(x)
    xint1 = (x_w>=-x0)*(x_w<x0)
    xint2 = (x_w>=x0)*(x_w<(1-x0))
    xint3 = (x_w>=(1-x0))*(x_w<(1+x0))
    y[xint1] = 1/3*a*(x_w[xint1])**3
    y[xint2] = 1/2*(x_w[xint2]-0.5)**2+d/c*np.cos(c*(x_w[xint2]-0.5)) + e*x_w[xint2]
    y[xint2] = y[xint2] - 1/2*(x0-0.5)**2 - d/c*np.cos(c*(x0-0.5)) - e*x0 + 1/3*a*x0**3
    y[xint3] = -1/3*a*(x_w[xint3]-1)**3 + x_w[xint3]
    y[xint3] =  y[xint3] + 1/3*a*(-x0)**3 - (1-x0) + 1/3*a*x0**3 + e*(1-x0) - e*x0 
    
    T = 1/3*a*(-x0)**3 - (1-x0) + 1/3*a*x0**3 + e*(1-x0) - e*x0 + 1
    period_num = np.floor(x).astype(int)
    y = y * (-np.ones(x.shape))**(period_num)
    offset1 = ( (period_num % 2) == 1 ) * T
    offset2 = period_num * T
    y = y + offset1 + offset2
    y = y - 0.5*x

    return y

class SmoothStairsModulation(Modulation):
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        if meas_prop.get_modulation_type() != 'smoothstairs':
            raise ValueError('measurement property is not consistent with the class')
        self.meas_prop = copy.deepcopy(meas_prop)
        self.sample_rate = meas_prop.get_sample_rate()
        self.Tchirp = meas_prop.get_chirp_length()
        self.bandwidth = meas_prop.get_bandwidth()
        self.linewidth = meas_prop.get_linewidth()
        self.lambd_c = meas_prop.get_carrier_wavelength()

        self.stair_func_params = {}
        c = 30
        e = 0.5
        x0 = 1/10
        d = ( (x0-0.5)+e-1/2*x0 ) / ( np.sin(c*(x0-0.5)) - 0.5*x0*c*np.cos(c*(x0-0.5)) )
        a = (1- c*d*np.cos(c*(x0-0.5))) / (2*x0)
        self.stair_func_params['c'] = c
        self.stair_func_params['e'] = e
        self.stair_func_params['x0'] = x0
        self.stair_func_params['d'] = d
        self.stair_func_params['a'] = a

    def generate_freq(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                     compute_jacobian: bool=False, normalize_freq: bool=True, aliased: bool=True):
     
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
        if compute_jacobian:
            if_tx = self.bandwidth * _stairfunc(t/self.Tchirp, **self.stair_func_params, compute_deriv=False)
            if_rx, if_rx_deriv = _stairfunc((t-delay)/self.Tchirp, **self.stair_func_params, compute_deriv=True)
            if_rx = self.bandwidth * if_rx - doppler_shift
            if_rx_deriv = self.bandwidth * if_rx_deriv / self.Tchirp
        else:
            if_tx = self.bandwidth * _stairfunc(t/self.Tchirp, **self.stair_func_params, compute_deriv=False)
            if_rx = self.bandwidth * _stairfunc((t-delay)/self.Tchirp, **self.stair_func_params, compute_deriv=False) - doppler_shift

        if aliased:
            if self.meas_prop.is_complex_available():
                mixed_if = utils.wrap(if_tx - if_rx, -self.sample_rate/2, self.sample_rate/2)
                aliased_deriv = 1
            else:
                mixed_if, aliased_deriv = utils.mirror_deriv(if_tx - if_rx, 0, self.sample_rate/2)
        else:
            mixed_if = if_tx - if_rx
            aliased_deriv = 1

        if compute_jacobian:
            d_deriv = aliased_deriv * if_rx_deriv * 2 / 3e8
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

    def generate_freq_x(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_jacobian: bool=False, normalize_freq: bool=True):
        
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
        if_rx = self.bandwidth * _stairfunc((t-delay)/self.Tchirp, **self.stair_func_params, compute_deriv=False) - doppler_shift
        if_rx += 0.5 * self.bandwidth

        if normalize_freq:
            if_rx /= self.sample_rate

        if compute_jacobian:
            raise NotImplementedError('compute jacobian for sinusoidal modulation generate_freq_x is not supported')

        return if_rx
    
    def generate_phase_x(self, t: np.ndarray, 
                         distance: float|np.ndarray=0, velocity: float|np.ndarray=0, 
                         compute_jacobian: bool=False, compute_hessian: bool=False):
        
        if compute_hessian:
            raise NotImplementedError()
    
        if isinstance(distance, np.ndarray):
            distance = np.expand_dims(distance, 1)
        if isinstance(velocity, np.ndarray):
            velocity = np.expand_dims(velocity, 1)

        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        v = 2*velocity/3e8
        f0 = 3e8 / self.lambd_c - self.bandwidth / 2

        phase = self.bandwidth*self.Tchirp * _stairfunc_integral((t-delay-v*t)/self.Tchirp, **self.stair_func_params)
        phase += self.bandwidth/2*(t-delay-v*t) - f0*v*t
        phase *= 2*np.pi

        if compute_jacobian:
            if isinstance(distance, np.ndarray):
                jacobian = np.zeros((len(distance), len(t), 2))
                jacobian[:,:,0] = self.bandwidth*_stairfunc((t-delay-v*t)/self.Tchirp, **self.stair_func_params)*(-2/3e8)
                jacobian[:,:,0] += self.bandwidth/2*(-2/3e8)
                jacobian[:,:,1] = self.bandwidth*_stairfunc((t-delay-v*t)/self.Tchirp, **self.stair_func_params)*(-2*t/3e8) 
                jacobian[:,:,1] += self.bandwidth/2*(-2*t/3e8) - f0*(2*t/3e8)
            else:
                jacobian = np.zeros((len(t), 2))
                jacobian[:,0] = self.bandwidth*_stairfunc((t-delay-v*t)/self.Tchirp, **self.stair_func_params)*(-2/3e8)
                jacobian[:,0] += self.bandwidth/2*(-2/3e8)
                jacobian[:,1] = self.bandwidth*_stairfunc((t-delay-v*t)/self.Tchirp, **self.stair_func_params)*(-2*t/3e8) 
                jacobian[:,1] += self.bandwidth/2*(-2*t/3e8) - f0*(2*t/3e8)
            jacobian *= 2*np.pi

            return phase, jacobian

        return phase
    
    def generate_phase(self, t: np.ndarray, 
                       distance: float|np.ndarray=0, velocity: float|np.ndarray=0, 
                       compute_jacobian: bool=False, compute_hessian: bool=False):
        
        phase_tx = self.generate_phase_x(t, 0, 0)
        if isinstance(distance, np.ndarray):
            phase_tx = np.expand_dims(phase_tx, axis=0)

        if compute_hessian:
            phase_rx, jacobian_rx, hessian_rx = self.generate_phase_x(t, distance, velocity, compute_hessian=True)
            
            return phase_tx-phase_rx,  -jacobian_rx, -hessian_rx
        
        if compute_jacobian:
            phase_rx, jacobian_rx = self.generate_phase_x(t, distance, velocity, compute_jacobian=True)
            
            return phase_tx-phase_rx, -jacobian_rx
        
        phase_rx = self.generate_phase_x(t, distance, velocity)
        return phase_tx-phase_rx


def _hardsteep(x,interval,compute_deriv=False):
    x_w = utils.mirror(x, 0, 1)
    y = np.zeros_like(x_w)
    xint1 = (x_w<0.5-interval/2)
    xint2 = (x_w>=0.5-interval/2)*(x_w<=0.5+interval/2)
    xint3 = (x_w>0.5+interval/2)
    y[xint1] = -0.5
    y[xint2] = (x_w[xint2]-0.5)/interval
    y[xint3] = 0.5

    if compute_deriv:
        z = np.zeros_like(x)
        z[xint1] = 0
        z[xint2] = 1/interval
        z[xint3] = 0
        period_num = np.floor(x).astype(int)
        z = z * (-np.ones(z.shape))**(period_num)
        return y, z
    return y

def _hardsteep_integral(x,interval):
    
    x_w = utils.mirror(x, 0, 1)
    y = np.zeros_like(x_w)
    xint1 = (x_w<0.5-interval/2)
    xint2 = (x_w>=0.5-interval/2)*(x_w<=0.5+interval/2)
    xint3 = (x_w>0.5+interval/2)

    y[xint2] = 1/2/interval*(x_w[xint2]-0.5)**2 + 0.5*interval/4
    y[xint3] = interval/4 + (x_w[xint3] - 0.5 - interval/2)
    
    T = interval/4 + (1-0.5-interval/2)
    period_num = np.floor(x).astype(int)
    y = y * (-np.ones(x.shape))**(period_num)
    offset1 = ( (period_num % 2) == 1 ) * T
    offset2 = period_num * T
    y = y + offset1 + offset2
    y = y - 0.5*x

    return y

class HardSteepModulation(Modulation):
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        if meas_prop.get_modulation_type() != 'hardsteep':
            raise ValueError('measurement property is not consistent with the class')
        self.meas_prop = copy.deepcopy(meas_prop)
        self.sample_rate = meas_prop.get_sample_rate()
        self.Tchirp = meas_prop.get_chirp_length()
        self.bandwidth = meas_prop.get_bandwidth()
        self.linewidth = meas_prop.get_linewidth()
        self.lambd_c = meas_prop.get_carrier_wavelength()

        self.func_params = {}
        self.func_params['interval'] = 200/self.sample_rate/self.Tchirp
        # self.func_params['interval'] = 1

    def generate_freq(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                     compute_jacobian: bool=False, normalize_freq: bool=True, aliased: bool=True):
     
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
        if compute_jacobian:
            if_tx = self.bandwidth * _hardsteep(t/self.Tchirp, **self.func_params, compute_deriv=False)
            if_rx, if_rx_deriv = _hardsteep((t-delay)/self.Tchirp, **self.func_params, compute_deriv=True)
            if_rx = self.bandwidth * if_rx - doppler_shift
            if_rx_deriv = self.bandwidth * if_rx_deriv / self.Tchirp
        else:
            if_tx = self.bandwidth * _hardsteep(t/self.Tchirp, **self.func_params, compute_deriv=False)
            if_rx = self.bandwidth * _hardsteep((t-delay)/self.Tchirp, **self.func_params, compute_deriv=False) - doppler_shift

        if aliased:
            if self.meas_prop.is_complex_available():
                mixed_if = utils.wrap(if_tx - if_rx, -self.sample_rate/2, self.sample_rate/2)
                aliased_deriv = 1
            else:
                mixed_if, aliased_deriv = utils.mirror_deriv(if_tx - if_rx, 0, self.sample_rate/2)
        else:
            mixed_if = if_tx - if_rx
            aliased_deriv = 1

        if compute_jacobian:
            d_deriv = aliased_deriv * if_rx_deriv * 2 / 3e8
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

    def generate_freq_x(self, t: np.ndarray, distance: float|Sequence|np.ndarray, velocity: float|Sequence|np.ndarray, 
                         compute_jacobian: bool=False, normalize_freq: bool=True):
        
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
        if_rx = self.bandwidth * _hardsteep((t-delay)/self.Tchirp, **self.func_params, compute_deriv=False) - doppler_shift
        if_rx += 0.5 * self.bandwidth
        
        if normalize_freq:
            if_rx /= self.sample_rate

        if compute_jacobian:
            raise NotImplementedError('compute jacobian for sinusoidal modulation generate_freq_x is not supported')

        return if_rx
    
    def generate_phase_x(self, t: np.ndarray, 
                         distance: float|np.ndarray=0, velocity: float|np.ndarray=0, 
                         compute_jacobian: bool=False, compute_hessian: bool=False):
        
        if compute_hessian:
            raise NotImplementedError()
    
        if isinstance(distance, np.ndarray):
            distance = np.expand_dims(distance, 1)
        if isinstance(velocity, np.ndarray):
            velocity = np.expand_dims(velocity, 1)

        delay = distance * 2 / 3e8
        gamma = self.bandwidth / self.Tchirp
        v = 2*velocity/3e8
        f0 = 3e8 / self.lambd_c - self.bandwidth / 2

        phase = self.bandwidth*self.Tchirp * _hardsteep_integral((t-delay-v*t)/self.Tchirp, **self.func_params)
        phase += self.bandwidth/2*(t-delay-v*t) - f0*v*t
        phase *= 2*np.pi

        if compute_jacobian:
            if isinstance(distance, np.ndarray):
                jacobian = np.zeros((len(distance), len(t), 2))
                jacobian[:,:,0] = self.bandwidth*_hardsteep((t-delay-v*t)/self.Tchirp, **self.func_params)*(-2/3e8)
                jacobian[:,:,0] += self.bandwidth/2*(-2/3e8)
                jacobian[:,:,1] = self.bandwidth*_hardsteep((t-delay-v*t)/self.Tchirp, **self.func_params)*(-2*t/3e8) 
                jacobian[:,:,1] += self.bandwidth/2*(-2*t/3e8) - f0*(2*t/3e8)
            else:
                jacobian = np.zeros((len(t), 2))
                jacobian[:,0] = self.bandwidth*_hardsteep((t-delay-v*t)/self.Tchirp, **self.func_params)*(-2/3e8)
                jacobian[:,0] += self.bandwidth/2*(-2/3e8)
                jacobian[:,1] = self.bandwidth*_hardsteep((t-delay-v*t)/self.Tchirp, **self.func_params)*(-2*t/3e8) 
                jacobian[:,1] += self.bandwidth/2*(-2*t/3e8) - f0*(2*t/3e8)
            jacobian *= 2*np.pi

            return phase, jacobian

        return phase
    
    def generate_phase(self, t: np.ndarray, 
                       distance: float|np.ndarray=0, velocity: float|np.ndarray=0, 
                       compute_jacobian: bool=False, compute_hessian: bool=False):
        
        phase_tx = self.generate_phase_x(t, 0, 0)
        if isinstance(distance, np.ndarray):
            phase_tx = np.expand_dims(phase_tx, axis=0)

        if compute_hessian:
            phase_rx, jacobian_rx, hessian_rx = self.generate_phase_x(t, distance, velocity, compute_hessian=True)
            
            return phase_tx-phase_rx,  -jacobian_rx, -hessian_rx
        
        if compute_jacobian:
            phase_rx, jacobian_rx = self.generate_phase_x(t, distance, velocity, compute_jacobian=True)
            
            return phase_tx-phase_rx, -jacobian_rx
        
        phase_rx = self.generate_phase_x(t, distance, velocity)
        return phase_tx-phase_rx

    
def get_modulation(meas_prop: FMCWMeasurementProperties):
    modulation = meas_prop.get_modulation_type()
    if modulation == 'triangle':
        return TriangularModulation(meas_prop)
    elif modulation == 'sinusoidal':
        return SinusoidalModulation(meas_prop)
    elif modulation == 'smoothstairs':
        return SmoothStairsModulation(meas_prop)
    elif modulation == 'hardsteep':
        return HardSteepModulation(meas_prop)
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
