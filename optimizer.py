
import numpy as np
import copy
import heapq
import time

class ObjectiveFunction():
    def __init__(self):
        pass
    def evaluate(self, x):
        # output must be shape (N,)
        # where N is the number of points
        raise NotImplementedError()
    def compute_gradient(self, x: np.ndarray):
        # output must be shape (N, k)
        # where N is number of points, k is the dimension
        raise NotImplementedError()
    def evaluate_and_compute_gradient(self, x: np.ndarray):
        # must return evaluated cost (N, ) and gradient (N, k)
        raise NotImplementedError()

class Scheduler():
    def __init__(self):
        pass
    def get_dimensions(self, n):
        raise NotImplementedError()
    def get_params(self, it, x):
        # this method should return kwargs that would be 
        # passed to objective function evaluate/compute_gradient
        raise NotImplementedError()
    def ready_for_convergence(self, it, x):
        # return true if optimizers can determine to end the iterations
        raise NotImplementedError()
    def reset(self):
        raise NotImplementedError()

def gradient_descent(x_init: np.ndarray, objective: ObjectiveFunction, 
                     max_n_iter: int=100, stepsize: float=1e-3, track=False, 
                     xtol: float=1e-4, scheduler: Scheduler|None=None):
    
    assert (np.ndim(x_init) == 2)
    
    N = x_init.shape[0]
    k = x_init.shape[1]

    x = copy.deepcopy(x_init)
    not_converged = np.ones((N,), dtype=bool)
    if track:
        tracked_vals = np.zeros((N, k+1, max_n_iter+1))
        tracked_vals[:, :-1, 0] = x_init
        tracked_vals[:, -1, 0] = objective.evaluate(x_init)

    for it in range(max_n_iter):

        if scheduler is not None:
            scheduled_params = scheduler.get_params(it, x)
        else:
            scheduled_params = dict()

        x_new = np.copy(x)
        grad_f = objective.compute_gradient(x[not_converged], **scheduled_params)
        x_new[not_converged] = x[not_converged] - stepsize * grad_f

        if track:
            tracked_vals[not_converged,:-1,it+1] = x[not_converged]
            tracked_vals[not_converged,-1,it+1] = objective.evaluate(x[not_converged], **scheduled_params)
            if it > 0:
                tracked_vals[~not_converged,:,it+1] = tracked_vals[~not_converged,:,it]

        not_converged = np.any(np.abs(x_new-x)>xtol, axis=1)
        if scheduler is not None:
            not_converged = not_converged + (not scheduler.ready_for_convergence(it, x_new))
        if not np.any(not_converged):
            x = x_new
            if track:
                tracked_vals = tracked_vals[:,:,:it+1]
            break

        x = x_new

    x_cost = objective.evaluate(x, **scheduled_params)
    if track:
        return_vals = (x, x_cost, tracked_vals)
    else:
        return_vals = (x, x_cost)
    return return_vals

def gradient_descent_backtrack_linesearch(x_init: np.ndarray, objective: ObjectiveFunction, 
                                          alpha0: float=1, beta: float=0.5, c: float=1e-4, max_n_iter: int=500, 
                                          track: bool=False, xtol: float=1e-4, scheduler: Scheduler|None=None,
                                          get_n_evals: bool=False):

    assert (np.ndim(x_init) == 2)
    
    N = x_init.shape[0]
    k = x_init.shape[1]
    
    x = copy.deepcopy(x_init)
    not_converged = np.ones((N,), dtype=bool) 
    if track:
        tracked_vals = np.zeros((N, k+1, max_n_iter+1))
        tracked_vals[:, :-1, 0] = x_init
        tracked_vals[:, -1, 0] = objective.evaluate(x_init)

    n_evals = 0
    for it in range(max_n_iter):

        if scheduler is not None:
            scheduled_params = scheduler.get_params(it, x)
        else:
            scheduled_params = dict()

        try:
            fx, grad_fx = objective.evaluate_and_compute_gradient(x[not_converged], **scheduled_params)
        except NotImplementedError:
            fx = objective.evaluate(x[not_converged], **scheduled_params)
            grad_fx = objective.compute_gradient(x[not_converged], **scheduled_params)
        n_evals += np.sum(not_converged)
        
        grad_fx_norm = np.linalg.norm(grad_fx, axis=1, keepdims=True) # (N, 1)
        grad_fx_norm_square = np.square(grad_fx_norm) # (N, 1)
        d = - grad_fx / (grad_fx_norm+1e-4) # (N, k)
        alpha = np.ones((np.sum(not_converged), 1)) * alpha0
        fx_new = objective.evaluate(x[not_converged]+alpha*d, **scheduled_params)
        tmp = fx_new > (fx - c*alpha[:,0]*grad_fx_norm_square[:,0]) # (N,)

        while np.any(tmp):
            alpha[tmp] = beta*alpha[tmp]
            fx_new[tmp] = objective.evaluate(x[not_converged][tmp]+alpha[tmp]*d[tmp], **scheduled_params)
            n_evals += np.sum(tmp)
            tmp = fx_new > (fx - c*alpha[:,0]*grad_fx_norm_square[:,0])

        x_new = np.copy(x)
        x_new[not_converged] = x[not_converged]+alpha*d # (N, k)
        if track:
            tracked_vals[not_converged,:-1,it+1] = x_new[not_converged]
            tracked_vals[not_converged,-1,it+1] = fx_new
            if it > 0:
                tracked_vals[~not_converged,:,it+1] = tracked_vals[~not_converged,:,it]

        not_converged = np.any(np.abs(x_new-x)>xtol, axis=1)
        if scheduler is not None:
            not_converged = not_converged + (not scheduler.ready_for_convergence(it, x_new))
        if not np.any(not_converged):
            x = x_new
            if track:
                tracked_vals = tracked_vals[:,:,:it+1]
            break

        x = x_new

    x_cost = objective.evaluate(x, **scheduled_params)
    n_evals += len(x)
    if track:
        return_val = (x, x_cost, tracked_vals)
    elif get_n_evals:
        return_val = (x, x_cost, n_evals)
    else:
        return_val = (x, x_cost)
    return return_val

def gridsearch(objective: ObjectiveFunction, grid: np.ndarray, keep_n: int=1, return_grid_val: bool=False):
    grid_orig_shape = grid.shape
    dim = grid.shape[-1]
    grid = np.reshape(grid, (-1, dim))
    # grid_val = np.zeros(len(grid))
    # for i, d_v_pair in enumerate(grid):
    #     grid_val[i] = objective.evaluate( np.array([[d_v_pair[0], d_v_pair[1]]]) )
    grid_val = objective.evaluate(grid)
    if keep_n > 1:
        # idx_n_smallest = heapq.nsmallest(keep_n, grid_val)
        idx_n_smallest = heapq.nsmallest(keep_n, enumerate(grid_val), key=lambda x: x[1])
        idx_n_smallest = np.array(idx_n_smallest)[:,0]
        idx_n_smallest = idx_n_smallest.astype(int)
        x_hat = grid[idx_n_smallest]
        x_hat_cost = grid_val[idx_n_smallest]
    else:
        idx = np.argmin(grid_val)
        x_hat = grid[[idx]]
        x_hat_cost = grid_val[[idx]]
    if return_grid_val:
        return_val =  ( x_hat, x_hat_cost, np.reshape(grid_val, grid_orig_shape[:-1]) )
    else:
        return_val = (x_hat, x_hat_cost)
    return return_val


# def gradient_descent(x_init, grad_f, n_iter=100, stepsize=1e-3, f=None, track=False):
#     x = copy.deepcopy(x_init)
#     if track and f is None:
#         raise ValueError('if track, f must be provided')
#     if track:
#         log = np.zeros((n_iter, len(x)+1))
#     for it in range(n_iter):
#         x = x - stepsize * grad_f(x)
#         if track:
#             log[it,:-1] = x
#             log[it,-1] = f(x)
#     if track:
#         return x, log
#     return x

# def gradient_descent_backtrack_linesearch(x_init, f, grad_f, beta=0.5, c=1e-1, n_iter=100, track=False, xtol=None):
    
#     x = copy.deepcopy(x_init)
#     if track:
#         log = np.zeros((n_iter, len(x)+1))
#     for it in range(n_iter):
#         fx = f(x)
#         grad_fx = grad_f(x)
#         grad_fx_norm = np.linalg.norm(grad_fx)
#         d = - grad_fx / grad_fx_norm
#         alpha = 1
#         fx_new = f(x+alpha*d)
#         while fx_new > (fx - c*alpha*grad_fx_norm):
#             alpha = beta*alpha
#             fx_new = f(x+alpha*d)
#         x_new = x+alpha*d
#         if track:
#             log[it,:-1] = x_new
#             log[it,-1] = fx_new
#         if (xtol is not None):
#             if np.all(np.abs(x_new-x)<xtol):
#                 x = x_new
#                 if track:
#                     log = log[:it]
#                 break
#         x = x_new
#     if track:
#         return x, log
#     return x
