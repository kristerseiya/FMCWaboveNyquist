
import numpy as np
import copy
from typing import Sequence

import fmcw_sys
from fmcw_sys import FMCWMeasurement, FMCWMeasurementProperties
from . import lorentzian
from utils import MACHINE_EPSILON
import utils


class GridGenerator():
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        self.meas_prop = copy.deepcopy(meas_prop)

    def generate_optimal_lattice_grid(self, precaution_rate: float=0.9):
        
        d_range, v_range = self.meas_prop.get_range()

        B = self.meas_prop.get_bandwidth()
        sample_rate = self.meas_prop.get_sample_rate()
        T = self.meas_prop.get_chirp_length()
        
        if self.meas_prop.get_modulation_type() == 'triangle':
            d_interval = sample_rate/2/B*T*2*3e8/2
            if not self.meas_prop.is_complex_available():
                d_interval = d_interval / 2
        elif self.meas_prop.get_modulation_type() == 'sinusoidal':
            d_interval = np.arcsin(sample_rate/2/B)*T/np.pi*2*3e8/2*2
            if not self.meas_prop.is_complex_available():
                d_interval = d_interval / 2
        elif self.meas_prop.get_modulation_type() == 'smoothstairs':
            d_interval = 120 * (sample_rate/B*T) / (4/5*1e-6)
            if not self.meas_prop.is_complex_available():
                d_interval = d_interval / 2
        elif self.meas_prop.get_modulation_type() == 'hardsteep':
            d_interval = 120 * (sample_rate/B*T) / (4/5*1e-6)
            if not self.meas_prop.is_complex_available():
                d_interval = d_interval / 2
        else:
            raise ValueError('lattice generation only supported for triangular and sinusoidal modulation')

        # precaution_rate = 0.9
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
                    # dv_grid1 = np.stack([d_grid, np.zeros_like(d_grid)], axis=1)
                    # dv_grid2 = np.stack([d_grid_shift, np.ones_like(d_grid_shift)*max_v], axis=1)
                    dv_grid1 = np.stack([d_grid, -np.ones_like(d_grid)*max_v/2], axis=1)
                    dv_grid2 = np.stack([d_grid_shift, np.ones_like(d_grid_shift)*max_v/2], axis=1)
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
    
    def generate_lattice_grid(self, d_n, v_n=1):
        
        if self.meas_prop.boundary_constraint:
            raise NotImplementedError('lattice generation only supported for no boundary constraint')

        d_range, v_range = self.meas_prop.get_range()

        B = self.meas_prop.get_bandwidth()
        sample_rate = self.meas_prop.get_sample_rate()
        T = self.meas_prop.get_chirp_length()
        
        d_interval = (d_range[-1] - d_range[0])/d_n
        v_interval = (v_range[-1] - v_range[0])/v_n
        
        d_grid = np.arange(d_range[0], d_range[-1]-MACHINE_EPSILON, d_interval)
        d_grid_shift = np.arange(d_range[0]+d_interval/2, d_range[-1]-MACHINE_EPSILON, d_interval)
        v_grid = np.arange(v_range[0], v_range[-1]-MACHINE_EPSILON, v_interval)
        # D_grid, V_grid = np.meshgrid(d_grid, v_grid)
        dv_grid = np.zeros((len(d_grid)*(len(v_grid)-len(v_grid)//2)+len(d_grid_shift)*len(v_grid)//2, 2))
        k = 0
        for i in range(len(v_grid)):
            if i % 2 == 0:
                dv_grid[k:k+len(d_grid), 0] = d_grid
                dv_grid[k:k+len(d_grid), 1] = v_grid[i]
                k += len(d_grid)
            else:
                dv_grid[k:k+len(d_grid_shift), 0] = d_grid_shift
                dv_grid[k:k+len(d_grid_shift), 1] = v_grid[i]
                k += len(d_grid_shift)
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
            res1, success = lorentzian.lorentzian_fitting(t[meas1_idx], signal[meas1_idx], shot_var=shot_var, meas_prop=self.meas_prop)
            if not success:
                return None
            freq1 = res1[0] / sample_rate
            if not assume_zero_velocity:
                res2, success = lorentzian.lorentzian_fitting(t[meas2_idx], signal[meas2_idx], shot_var=shot_var, meas_prop=self.meas_prop)
                if not success:
                    return None
                freq2 = res2[0] / sample_rate
        elif method == 'maxpd_fine':
            res1, success = lorentzian.max_periodogram(signal[meas1_idx], method='fine', meas_prop=self.meas_prop)
            if not success:
                return None
            freq1 = res1
            if not assume_zero_velocity:
                res2, success = lorentzian.max_periodogram(signal[meas2_idx], method='fine', meas_prop=self.meas_prop)
                if not success:
                    return None
                freq2 = res2 / sample_rate
        elif method == 'maxpd_coarse':
            res1, _ = lorentzian.max_periodogram(signal[meas1_idx], method='coarse', meas_prop=self.meas_prop)
            freq1 = res1
            if not assume_zero_velocity:
                res2, _ = lorentzian.max_periodogram(signal[meas2_idx], method='coarse', meas_prop=self.meas_prop)
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
            beatfreq, _ = lorentzian.max_periodogram(mixed_signal, method='coarse')
        elif method == 'maxpd_fine':
            beatfreq, success = lorentzian.max_periodogram(mixed_signal, method='fine')
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