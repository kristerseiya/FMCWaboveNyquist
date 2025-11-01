
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft, ifft
from typing import Sequence
import copy

import fmcw_sys
from fmcw_sys import FMCWMeasurement, FMCWMeasurementProperties
from .estimators_base import Estimator
from . import lorentzian
import optimizer
from optimizer import ObjectiveFunction, Scheduler
from utils import MACHINE_EPSILON
import utils
from .grid import GridGenerator
import time
from scipy.optimize import minimize
from functools import partial

class IFLikelihood():
    def __init__(self, meas_prop: FMCWMeasurementProperties):
        self.meas_prop = copy.deepcopy(meas_prop)
    def evaluate_nll(self):
        raise NotImplementedError()
    def compute_nll_gradient(self):
        raise NotImplementedError()
    def nll_grid_search(self):
        raise NotImplementedError()


class IFWrappedNormalLikelihood(IFLikelihood, ObjectiveFunction):
    def __init__(self, meas_prop: FMCWMeasurementProperties, K=2, eps=1e-20):
        self.meas_prop = copy.deepcopy(meas_prop)
        self.if_generator = fmcw_sys.modulation.get_modulation(meas_prop)
        self.linewidth = self.meas_prop.get_linewidth()
        self.sample_rate = self.meas_prop.get_sample_rate()
        self.if_var = self.linewidth*2/self.sample_rate/(2*np.pi)
        self.K = K
        self.snr_adjust_polyfit_coefs = np.load(os.path.join(this_dir,'project_normal_diff_var_polyfit.npy'))
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
                adjust_var_deriv[(snr_dB>=-20)*(snr_dB<=30)] *= utils.compute_polynomial_derivative(snr_dB[(snr_dB>=-20)*(snr_dB<=30)], self.snr_adjust_polyfit_coefs) / (2*np.pi**2)
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
                    adjust_var_deriv = adjust_var_deriv * utils.compute_polynomial_derivative(snr_dB, self.snr_adjust_polyfit_coefs) / (2*np.pi**2)
                return adjust_var, adjust_var_deriv
            return adjust_var

    def adjust_for_snr(self, snr, dB=True):
        self.if_var = self.linewidth*2/self.sample_rate/(2*np.pi) + self.snr_adjust_func(snr, dB=dB)

    def remove_snr_adjustment(self):
        self.if_var = self.linewidth*2/self.sample_rate/(2*np.pi)
    
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
            if_hat = self.if_generator.generate_freq(self.t, x[:,0], 0, compute_jacobian=False, normalize_freq=True, aliased=True)
        else:
            if_hat = self.if_generator.generate_freq(self.t, x[:,0], x[:,1], compute_jacobian=False, normalize_freq=True, aliased=True)

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
            if_hat, d_deriv, v_deriv = self.if_generator.generate_freq(self.t, x[:,0], 0, compute_jacobian=True, normalize_freq=True, aliased=True)
        else:
            if_hat, d_deriv, v_deriv = self.if_generator.generate_freq(self.t, x[:,0], x[:,1], compute_jacobian=True, normalize_freq=True, aliased=True)

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
            if_hat, d_deriv, v_deriv = self.if_generator.generate_freq(self.t, x[:,0], 0, compute_jacobian=True, normalize_freq=True, aliased=True)
        else:
            if_hat, d_deriv, v_deriv = self.if_generator.generate_freq(self.t, x[:,0], x[:,1], compute_jacobian=True, normalize_freq=True, aliased=True)
        
        Z = np.zeros((len(self.t)))
        denom = np.zeros((len(self.t)))
        numer = np.zeros((len(self.t)))
        for k in np.arange(-self.K, self.K+1):
            Zk = np.exp(-(self.ifx-if_hat+k)**2/(2*(self.if_var+extra_var)))
            Z = Z + Zk
            denom = denom + Zk
            numer = numer + Zk * ( (self.ifx-if_hat+k)/(self.if_var+extra_var) )
        nll = - np.sum( np.log(Z+MACHINE_EPSILON), axis=1) + len(self.t)/2*np.log(2*(self.if_var+extra_var))
        tmp = - numer / (denom + MACHINE_EPSILON)
        d_deriv = np.sum(tmp * d_deriv, axis=1)
        v_deriv = np.sum(tmp * v_deriv, axis=1)
        if self.meas_prop.is_zero_velocity():
            gradient = np.stack([d_deriv, np.zeros_like(v_deriv)], axis=1)
        else:
            gradient = np.stack([d_deriv, v_deriv], axis=1)
        return nll, gradient
    
    def create_scheduler(self):
        return ExtraVarScheduler(self.if_var, init_var=0.1, m=300)

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
        if it >= self._m:
            extra_var = 0
        else:
            extra_var = ( np.sqrt(self._init_var) / self._m * (self._m - it) + np.sqrt(self._correct_var) / self._m * (it) )**2 - self._correct_var
        return {'extra_var': extra_var}
    
    def ready_for_convergence(self, it: int, x: np.ndarray):
        return it >= self._m

    def reset(self):
        self.curr_x = None
        self._curr_var = self._init_var
        self._ready_for_converge = (self._correct_var >= self._init_var)

class IFShortestPath(IFLikelihood, ObjectiveFunction):
    def __init__(self, meas_prop: FMCWMeasurementProperties, K=2, eps=1e-20):
        self.meas_prop = copy.deepcopy(meas_prop)
        self.if_generator = fmcw_sys.modulation.get_modulation(meas_prop)
        self.linewidth = self.meas_prop.get_linewidth()
        self.sample_rate = self.meas_prop.get_sample_rate()
        self.if_var = self.linewidth*2/self.sample_rate/(2*np.pi)
        self.K = K
        self.snr_adjust_polyfit_coefs = np.load(os.path.join(this_dir, '..', 'phase_noise','project_normal_diff_var_polyfit.npy'))
        self.snr_adjust_polyfit_func = np.poly1d(self.snr_adjust_polyfit_coefs)
        self.eps = eps
        self.t = None
        self.ifx = None
    
    def store_observations(self, t: np.ndarray, ifx: np.ndarray):

        assert (np.ndim(t) == 1) and (np.ndim(ifx) == 1)
        
        self.t = copy.deepcopy(t)
        self.ifx = copy.deepcopy(ifx)

    def evaluate(self, x: np.ndarray, extra_var: float|None=None):

        if (np.ndim(x) != 2) or (x.shape[-1] != 2):
            raise ValueError("Dimension of x not acceptable")

        if extra_var is None:
            extra_var = 0

        if self.meas_prop.is_zero_velocity():
            if_hat = self.if_generator.generate_freq(self.t, x[:,0], 0, compute_jacobian=False, normalize_freq=True, aliased=True)
        else:
            if_hat = self.if_generator.generate_freq(self.t, x[:,0], x[:,1], compute_jacobian=False, normalize_freq=True, aliased=True)

        shortest_path_length = np.sum((np.mod(self.ifx-if_hat+0.5,1)-0.5)**2, axis=1)
        return shortest_path_length
    
    def compute_gradient(self, x: np.ndarray):

        if (np.ndim(x) != 2) or (x.shape[-1] != 2):
            raise ValueError("Dimension of x not acceptable")

        if self.meas_prop.is_zero_velocity():
            if_hat, d_deriv, v_deriv = self.if_generator.generate_freq(self.t, x[:,0], 0, compute_jacobian=True, normalize_freq=True, aliased=True)
        else:
            if_hat, d_deriv, v_deriv = self.if_generator.generate_freq(self.t, x[:,0], x[:,1], compute_jacobian=True, normalize_freq=True, aliased=True)

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
            if_hat, d_deriv, v_deriv = self.if_generator.generate_freq(self.t, x[:,0], 0, compute_jacobian=True, normalize_freq=True, aliased=True)
        else:
            if_hat, d_deriv, v_deriv = self.if_generator.generate_freq(self.t, x[:,0], x[:,1], compute_jacobian=True, normalize_freq=True, aliased=True)
        
        shortest_path_length = np.sum((np.mod(self.ifx-if_hat+0.5,1)-0.5)**2, axis=1)
        tmp = 2*(np.mod(if_hat-self.ifx+0.5,1)-0.5)
        d_deriv = np.sum(tmp * d_deriv, axis=1)
        v_deriv = np.sum(tmp * v_deriv, axis=1)
        if self.meas_prop.is_zero_velocity():
            gradient = np.stack([d_deriv, np.zeros_like(v_deriv)], axis=1)
        else:
            gradient = np.stack([d_deriv, v_deriv], axis=1)
        return shortest_path_length, gradient

class IFRegressor(Estimator):
    def __init__(self, meas_prop: FMCWMeasurementProperties, 
                 gridgen_type='optimal', init_step='none', method='scipy',
                 likelihood='wrapped_normal', ignore_quadrature=False, keep_n_from_gridsearch=3,
                 snr_adjustment=True, gridgen_param=None, gd_max_n_iter=500,
                 grid_d_num=10, grid_v_num=1,
                 average=False):
        self.meas_prop = copy.deepcopy(meas_prop)
        self.ignore_quadrature = ignore_quadrature
        if self.ignore_quadrature:
            self.meas_prop.complex_available = False
        if likelihood == 'wrapped_normal':
            self.if_likelihood = IFWrappedNormalLikelihood(self.meas_prop)
        elif likelihood == 'shortest_path':
            self.if_likelihood = IFShortestPath(self.meas_prop)
        self.grid_generator = GridGenerator(self.meas_prop) 
        self.gridgen_type = gridgen_type
        self.gridgen_param = gridgen_param  
        self.snr_adjustment = snr_adjustment
        self.gd_max_n_iter = gd_max_n_iter
        self.keep_n_from_gridsearch = keep_n_from_gridsearch
        self.init_step = init_step
        self.method = method
        self.assume_zero_velocity = self.meas_prop.is_zero_velocity()
        self.test_flag = False
        self.average = average
        self.grid_d_num = grid_d_num
        self.grid_v_num = grid_v_num
        
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

        if not isinstance(self.if_likelihood, IFShortestPath):
            if  self.snr_adjustment and (second_output is not None):
                snr_estimate = 10*np.log10((np.var(mixed_signal)-np.var(second_output))/np.var(second_output))
                self.if_likelihood.adjust_for_snr(snr_estimate)
            else:
                self.if_likelihood.remove_snr_adjustment()

        
        T = self.meas_prop.Tchirp
        sample_rate = self.meas_prop.sample_rate
        M = int(np.ceil(2*T*sample_rate))
        N = len(mixed_signal)
        L = int(np.floor(N/M))
        if self.average:
            idx_partitions = [(i*M, (i+1)*M) for i in range(L)]
            x_hat_finals = np.zeros((L,2))
        else:
            idx_partitions = [(0, N)]
            x_hat_finals = np.zeros((1,2))

        for k, idx_part in enumerate(idx_partitions):
            t_, mixed_signal_ = t[idx_part[0]:idx_part[1]], mixed_signal[idx_part[0]:idx_part[1]]
            tt, ifx = utils.estimate_if(t_, mixed_signal_, method='polar_discriminator', mirror=False)
            self.if_likelihood.store_observations(tt, ifx)
            
            if self.test_flag:
                intermediate_results['if_estimate'] = (tt, ifx)
                intermediate_results['likelihood_model'] = copy.deepcopy(self.if_likelihood)

            now = time.time()
            if self.gridgen_type == 'optimal':
                grid = self.grid_generator.generate_optimal_lattice_grid(precaution_rate=0.9)
            elif self.gridgen_type == 'uniform':
                grid = self.grid_generator.generate_lattice_grid(self.grid_d_num, self.grid_v_num)
            elif self.gridgen_type == 'smart_maxpd':
                if self.meas_prop.get_modulation_type() != 'triangle':
                    raise NotImplementedError('smart grid generation is only for triangular modulation')
                grid = self.grid_generator.generate_smart_grid(t_, mixed_signal_, second_output, method='maxpd_coarse')
            elif self.gridgen_type == 'smart_lorentzian':
                if self.meas_prop.get_modulation_type() != 'triangle':
                    raise NotImplementedError('smart grid generation is only for triangular modulation')
                grid = self.grid_generator.generate_smart_grid(t_, mixed_signal_, second_output, method='lorentzian')
            else:
                raise RuntimeError("did not recognize gridgen_type")

            if self.test_flag:
                intermediate_results['grid'] = copy.deepcopy(grid)
                
            if self.init_step == 'gridsearch':
                x_init, x_init_cost = optimizer.gridsearch(self.if_likelihood, grid, 
                                                            keep_n=self.keep_n_from_gridsearch, return_grid_val=False)
            elif self.init_step == 'shortestpath_likelihood':
                x_init_init = np.reshape(grid, (-1,2))
                init_if_likelihood = IFShortestPath(self.meas_prop)
                init_if_likelihood.store_observations(tt, ifx)
                x_init, x_init_cost = optimizer.gradient_descent_backtrack_linesearch(x_init_init, init_if_likelihood, 
                                                                                    max_n_iter=self.gd_max_n_iter,
                                                                                    alpha0=1, beta=0.1, c=1e-1,
                                                                                    xtol=1e-4, track=False)
            elif self.init_step == 'none':
                x_init = np.reshape(grid, (-1, 2))
                x_init_cost = self.if_likelihood.evaluate(x_init)
                if self.test_flag:
                    grid_val = np.reshape(x_init_cost, grid.shape[:-1])
                    
            elif self.init_step == "subsample":
                
                x_init_init = np.reshape(grid, (-1,2))
                B = self.meas_prop.get_bandwidth()
                fs = self.meas_prop.get_sample_rate()
                subsamp_K = int(np.floor(len(ifx) / (int(np.ceil(B/fs)) * 10)))
                init_if_likelihood = IFShortestPath(self.meas_prop)
                init_if_likelihood.store_observations(tt[::subsamp_K], ifx[::subsamp_K])
                x_init, x_init_cost = optimizer.gradient_descent_backtrack_linesearch(x_init_init, init_if_likelihood, 
                                                                                                max_n_iter=self.gd_max_n_iter, 
                                                                                                alpha0=1, beta=0.1, c=1e-1,
                                                                                                xtol=1e-4, track=False)
            
            if self.test_flag:
                intermediate_results['x_init'] = copy.deepcopy(x_init)
                intermediate_results['x_init_cost'] = copy.deepcopy(x_init_cost)

            if self.method == 'gradient_descent':
                
                
                
                if self.init_step != 'shortestpath_likelihood' and self.init_step != 'subsampled_if':
                    try:
                        scheduler = self.if_likelihood.create_scheduler()
                    except AttributeError:
                        scheduler = None 
                else:
                    scheduler = None
                if not self.test_flag:
                    x_hat, x_hat_cost = optimizer.gradient_descent_backtrack_linesearch(x_init, self.if_likelihood, 
                                                                                        max_n_iter=self.gd_max_n_iter, 
                                                                                        xtol=1e-4, track=False,
                                                                                        alpha0=1, beta=0.1, c=1e-1,
                                                                                        scheduler=scheduler)
                else:
                    x_hat, x_hat_cost, gd_track = optimizer.gradient_descent_backtrack_linesearch(x_init, self.if_likelihood, 
                                                                                                max_n_iter=self.gd_max_n_iter, 
                                                                                                xtol=1e-4, track=True,
                                                                                                alpha0=1, beta=0.1, c=1e-1,
                                                                                                scheduler=scheduler)
                    intermediate_results['x_hat'] = copy.deepcopy(x_hat)
                    intermediate_results['x_hat_cost'] = copy.deepcopy(x_hat_cost)
                    intermediate_results['gd_track'] = gd_track
                x_hat_final = x_hat[np.argmin(x_hat_cost)]
                max_d = self.meas_prop.get_max_d()
                max_v = self.meas_prop.get_max_v()
                x_hat_final[0] = np.mod(x_hat_final[0], max_d)
                x_hat_final[1] = np.mod(x_hat_final[1]+max_v, 2*max_v)-max_v

            elif self.method == 'none':
                if x_init.shape[0] > 1:
                    x_hat_final = x_init[np.argmin(x_init_cost)]
                else:
                    x_hat_final = x_init[0]
                    
            elif self.method == "LBFGS":
                max_d = self.meas_prop.get_max_d()
                max_v = self.meas_prop.get_max_v()
                
                self.total_n_evals = 0
                grid = grid.reshape((-1, 2))
                def obj_func(x):
                    return self.if_likelihood.evaluate(np.expand_dims(x,axis=0), extra_var=0.1)[0]
                def jac(x):
                    return self.if_likelihood.compute_gradient(np.expand_dims(x,axis=0), extra_var=0.1)[0]
                xhats = np.zeros((len(grid),2))
                costs = np.zeros((len(grid),))
                for i, x0 in enumerate(grid):
                    res = minimize(obj_func, 
                                x0, jac=jac,
                                method="BFGS")
                    xhats[i,0] = np.mod(res.x[0], max_d)
                    xhats[i,1] = np.mod(res.x[1]+max_v, 2*max_v)-max_v
                    costs[i] = res.fun
                    self.total_n_evals += res.nit
                def obj_func(x):
                    return self.if_likelihood.evaluate(np.expand_dims(x,axis=0), extra_var=0)[0]
                def jac(x):
                    return self.if_likelihood.compute_gradient(np.expand_dims(x,axis=0), extra_var=0)[0]
                x_hat = xhats[np.argmin(costs)]
                res = minimize(obj_func, 
                                x_hat, jac=jac,
                                method="BFGS")
                x_hat_final = res.x
                x_hat_final[0] = np.mod(res.x[0], max_d)
                x_hat_final[1] = np.mod(res.x[1]+max_v, 2*max_v)-max_v
                self.total_n_evals += res.nit

            x_hat_finals[k] = x_hat_final

            x_hat_final = np.mean(x_hat_finals, axis=0)

        if not self.test_flag:
            return_val = x_hat_final
        else:
            return_val = (x_hat_final, intermediate_results)

        return return_val
    
    def estimate2(self, t: np.ndarray, mixed_signal: np.ndarray, second_output: np.ndarray|None=None):
        
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

        if not isinstance(self.if_likelihood, IFShortestPath):
            if  self.snr_adjustment and (second_output is not None):
                snr_estimate = 10*np.log10((np.var(mixed_signal)-np.var(second_output))/np.var(second_output))
                self.if_likelihood.adjust_for_snr(snr_estimate)
            else:
                self.if_likelihood.remove_snr_adjustment()

        
        T = self.meas_prop.Tchirp
        sample_rate = self.meas_prop.sample_rate
        M = int(np.ceil(2*T*sample_rate))
        N = len(mixed_signal)
        L = int(np.floor(N/M))
        if self.average:
            idx_partitions = [(i*M, (i+1)*M) for i in range(L)]
            x_hat_finals = np.zeros((L,2))
        else:
            idx_partitions = [(0, N)]
            x_hat_finals = np.zeros((1,2))

        for k, idx_part in enumerate(idx_partitions):
            t_, mixed_signal_ = t[idx_part[0]:idx_part[1]], mixed_signal[idx_part[0]:idx_part[1]]
            tt, ifx = utils.estimate_if(t_, mixed_signal_, method='polar_discriminator', mirror=False)
            self.if_likelihood.store_observations(tt, ifx)
            
            if self.test_flag:
                intermediate_results['if_estimate'] = (tt, ifx)
                intermediate_results['likelihood_model'] = copy.deepcopy(self.if_likelihood)

            if self.gridgen_type == 'optimal':
                grid = self.grid_generator.generate_optimal_lattice_grid()
            elif self.gridgen_type == 'uniform':
                grid = self.grid_generator.generate_lattice_grid(self.grid_d_num, self.grid_v_num)
            elif self.gridgen_type == 'smart_maxpd':
                if self.meas_prop.get_modulation_type() != 'triangle':
                    raise NotImplementedError('smart grid generation is only for triangular modulation')
                grid = self.grid_generator.generate_smart_grid(t_, mixed_signal_, second_output, method='maxpd_coarse')
            elif self.gridgen_type == 'smart_lorentzian':
                if self.meas_prop.get_modulation_type() != 'triangle':
                    raise NotImplementedError('smart grid generation is only for triangular modulation')
                grid = self.grid_generator.generate_smart_grid(t_, mixed_signal_, second_output, method='lorentzian')
            else:
                raise RuntimeError("did not recognize gridgen_type")
            
            grid = grid.reshape((-1, 2))
            def obj_func(x):
                return self.if_likelihood.evaluate(np.expand_dims(x,axis=0), extra_var=0.1)[0]
            def jac(x):
                return self.if_likelihood.compute_gradient(np.expand_dims(x,axis=0), extra_var=0.1)[0]
            xhats = np.zeros((len(grid),2))
            costs = np.zeros((len(grid),))
            for i, x0 in enumerate(grid):
                res = minimize(obj_func, 
                               x0, jac=jac,
                               method="BFGS")
                xhats[i] = res.x
                costs[i] = res.fun
            def obj_func(x):
                return self.if_likelihood.evaluate(np.expand_dims(x,axis=0), extra_var=0)[0]
            def jac(x):
                return self.if_likelihood.compute_gradient(np.expand_dims(x,axis=0), extra_var=0)[0]
            x_hat = xhats[np.argmin(costs)]
            res = minimize(obj_func, 
                            x_hat, jac=jac,
                            method="BFGS")
                
            x_hat_finals[k] = res.x
        
        x_hat_final = np.mean(x_hat_finals, axis=0)
        max_d = self.meas_prop.get_max_d()
        max_v = self.meas_prop.get_max_v()
        x_hat_final[0] = np.mod(x_hat_final[0], max_d)
        x_hat_final[1] = np.mod(x_hat_final[1]+max_v, 2*max_v)-max_v

        if not self.test_flag:
            return_val = x_hat_final
        else:
            return_val = (x_hat_final, intermediate_results)

        return return_val
            

    
    def set_test_flag(self, test: bool):
        self.test_flag = test
