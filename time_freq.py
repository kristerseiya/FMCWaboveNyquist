
import numpy as np
from scipy import signal, linalg, ndimage
import scipy.fftpack


def short_time_fourier_transform(x, window, stride=1, sample_rate=1, real=False, t0=0, zero_pad=64):
    
    N = len(x)
    if isinstance(window, int):
        window_size = window
        t = np.arange(-window_size//2, window_size-window_size//2)
        h = np.cos(np.pi*t/window_size)**2
    elif isinstance(window, np.ndarray):
        window_size = len(window)
        h = window
    else:
        raise ValueError("window must be either int or 1d numpy array")
    
    start_idx = np.arange(0,N+1-window_size,stride)
    tf_plot = np.zeros((window_size+zero_pad,len(start_idx)), dtype=np.complex128)
    for j, i in enumerate(start_idx):
        y = x[i:(i+window_size)] * h
        if zero_pad > 0:
            y = np.pad(y, (0, zero_pad), mode="constant", constant_values=0)
        tf_plot[:,j] =  scipy.fftpack.fft(y)
    negfreq = tf_plot[(window_size+zero_pad+1)//2:, :]
    posfreq = tf_plot[:(window_size+zero_pad+1)//2, :]
    if real:
        tf_plot = posfreq
    else:
        tf_plot = np.concatenate([negfreq, posfreq], axis=0)

    if real:
        stft_f = np.arange(tf_plot.shape[0]) / tf_plot.shape[0] * sample_rate / 2
    else:
        stft_f = np.arange(tf_plot.shape[0]) / tf_plot.shape[0] * sample_rate
        stft_f = stft_f - sample_rate/2
    stft_t = np.arange(tf_plot.shape[1]) * stride/sample_rate + window_size/sample_rate/2 + t0
    return stft_f, stft_t, tf_plot

def reassignment_method(x, window_length, hop_length, thres=0):
    t = np.arange(-window_length//2, window_length-window_length//2)
    h = np.cos(np.pi*t/window_length)**2
    dh = 2*np.cos(np.pi*t/window_length)*(-np.sin(np.pi*t/window_length))*np.pi/window_length
    f, t, sh = short_time_fourier_transform(x, h, hop_length, zero_pad=0)
    _, _, sdh = short_time_fourier_transform(x, dh, hop_length, zero_pad=0)

    ff = np.stack([f]*sh.shape[1], axis=1)
    valid = np.abs(sh) > thres
    newf = ff[valid]-np.imag(sdh[valid]/sh[valid]/2/np.pi)
    t = np.arange(sh.shape[1]).astype(float)
    t = np.stack([t]*sh.shape[0], axis=0)
    t[~valid] = np.nan
    f = np.zeros(sh.shape)
    f[:] = np.nan
    f[valid] = newf
    mags = np.abs(sh)
    mags[~valid] = np.nan
    return t, f, mags

def second_order_reassignment_method(x, window_length, hop_length, thres=0):
    t = np.arange(-window_length//2, window_length-window_length//2)
    h = np.cos(np.pi*t/window_length)**2
    dh = 2*np.cos(np.pi*t/window_length)*(-np.sin(np.pi*t/window_length))*np.pi/window_length
    th = t * h
    ddh = 2*(-np.sin(np.pi*t/window_length))*np.pi/window_length*(-np.sin(np.pi*t/window_length))*np.pi/window_length
    ddh = ddh + 2*np.cos(np.pi*t/window_length)*(-np.cos(np.pi*t/window_length))*np.pi/window_length*np.pi/window_length
    tdh = t * dh
    f, t, sh = short_time_fourier_transform(x, h, hop_length, zero_pad=0)
    _, _, sdh = short_time_fourier_transform(x, dh, hop_length)
    _, _, sth = short_time_fourier_transform(x, th, hop_length)
    _, _, sddh = short_time_fourier_transform(x, ddh, hop_length)
    _, _, stdh = short_time_fourier_transform(x, tdh, hop_length)

    ff = np.stack([f]*sh.shape[1], axis=1)
    tt = np.arange(sh.shape[1]).astype(float)
    tt = np.stack([tt]*sh.shape[0], axis=0)
    valid1 = np.abs(sh) > thres
    valid2 = valid1 * (np.abs(sth*sdh-stdh*sh) > thres)
    f_tilde_ = ff[valid1] - sdh[valid1]/sh[valid1]/(1j*2*np.pi)
    f_tilde = ff[valid2] - sdh[valid2]/sh[valid2]/(1j*2*np.pi)
    t_tilde = tt[valid2] + sth[valid2]/sh[valid2]
    q_tilde = (sddh[valid2]*sh[valid2]-(sdh[valid2])**2) / (sth[valid2]*sdh[valid2]-stdh[valid2]*sh[valid2]) / (1j*2*np.pi)
    
    tt[~valid1] = np.nan
    f = np.zeros(sh.shape)
    f[:] = np.nan
    f[valid1] = np.real(f_tilde_)
    f[valid2] = np.real(f_tilde + q_tilde * (tt[valid2] - t_tilde))
    mags = np.abs(sh)
    mags[~valid1] = np.nan
    return tt, f, mags

def ridge_detection(times, freqs, mags, gamma, c1, c2, n_survivor=7):
    max_idx_col = np.argmax(mags, 1)
    survivors = np.zeros((n_survivor, mags.shape[1]), dtype=int)
    path_dist = np.zeros((n_survivor,))

    # first
    valididx = np.arange(mags.shape[0])[~np.isnan(mags[:,0])]
    if (len(valididx)<n_survivor):
            print(len(valididx))
    dist_metric = - 20*np.log10(mags[valididx,0]+1e-10)
    bestidx = np.argpartition(dist_metric, n_survivor)[:n_survivor]
    survivors[:,0] = valididx[bestidx]
    path_dist = dist_metric[bestidx]

    # second
    valididx = np.arange(mags.shape[0])[~np.isnan(mags[:,1])]
    if (len(valididx)<n_survivor):
            print(len(valididx))
    diff1 = - np.reshape(freqs[survivors[:,0],0], (-1,1)) + freqs[valididx,1]
    diff1 /= (times[valididx[0],1]-times[valididx[0],0])
    dist_metric = np.reshape(path_dist, (-1,1)) - 20*np.log10(mags[valididx,1]+1e-10) + np.minimum(np.abs(diff1), np.abs(diff1-gamma))*c1
    bestidx = np.argpartition(dist_metric.flatten(), n_survivor)[:n_survivor]
    
    bestidx = np.unravel_index(bestidx, dist_metric.shape)
    new_survivors = np.zeros((n_survivor, 2))
    for n in range(n_survivor):
        new_survivors[n,:1] = survivors[bestidx[0][n],:1]
        new_survivors[n,-1] = valididx[bestidx[1][n]]
        path_dist[n] = dist_metric[bestidx[0][n], bestidx[1][n]]
    survivors[:,:2] = new_survivors

    # rest
    for i in range(2, mags.shape[1]):

        valididx = np.arange(mags.shape[0])[~np.isnan(mags[:,i])]
        if (len(valididx)<n_survivor):
            print(len(valididx))
        diff1 = - np.reshape(freqs[survivors[:,i-1],i-1], (-1,1)) + freqs[valididx,i]
        diff1 /= (times[valididx[0],i]-times[survivors[0,i-1],i-1])
        diff2 = ( freqs[survivors[:,i-2],i-2] - freqs[survivors[:,i-1],i-1] ) / ( times[survivors[0,i-2],i-2] - times[survivors[0,i-1],i-1] )
        diff2 = np.reshape(diff2,(-1,1)) + ( - np.reshape(freqs[survivors[:,i-1],i-1], (-1,1)) + freqs[valididx,i]  ) / ( - times[survivors[0,i-1],i-1] + times[valididx[0],i] )
        dist_metric = np.reshape(path_dist, (-1,1)) - 20*np.log10(mags[valididx,i]+1e-10) + np.minimum(np.abs(diff1), np.abs(np.abs(diff1)-gamma))*c1 #+ diff2**2*c2
        bestidx = np.argpartition(dist_metric.flatten(), n_survivor)[:n_survivor]
        bestidx = np.unravel_index(bestidx, dist_metric.shape)
        new_survivors = np.zeros((n_survivor, i+1))
        for n in range(n_survivor):
            new_survivors[n,:i] = survivors[bestidx[0][n],:i]
            new_survivors[n,-1] = valididx[bestidx[1][n]]
            path_dist[n] = dist_metric[bestidx[0][n], bestidx[1][n]]
        survivors[:,:(i+1)] = new_survivors
        

    k = np.argmin(path_dist)
    return survivors[k]

    







        
