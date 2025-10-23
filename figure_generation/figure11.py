
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import utils
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fmcw_sys

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
font = {'fontname':'Times New Roman'}
plt.rc('text.latex', preamble=r"\usepackage{bm}\boldmath\renewcommand{\seriesdefault}{\bfdefault}")
size_mult = 1

this_dir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(utils.PROJECT_DIR, 'sim', 'd_est_sim_results')
results_filekey = [('011625013845_estimates_011625013845_seeds_bunny4.npz','wn_snradjust_lattice',{'color':'purple', 'label':'Proposed ($T=2\mu s$, x10 periods)'}), 
                   ('022325035011_estimates_011625013845_seeds_bunny4.npz','maxpd',{'color':'dodgerblue', 'label':'Proposed ($T=20\mu s$, x1 period)'})]



cmap_name = 'jet_r'
cmap_name = 'turbo'
fig, ax = plt.subplots(1,6, figsize=(14*size_mult,3*size_mult))
fig.subplots_adjust(wspace=0.2, hspace=0.5, left=0.005, right=0.965, top=0.8)
cmap = plt.get_cmap(cmap_name)
true_depth = Image.open('bunnydepth4.png')
true_depth = np.array(true_depth).astype(float) - 290
vmax = np.amax(true_depth)
vmin = np.amin(true_depth)
print(vmax)
print(vmin)
depth_color = np.ones((*true_depth.shape, 3), dtype=true_depth.dtype)
depth_color[true_depth>0] = cmap((true_depth[true_depth>0]-vmin)/(vmax-vmin))[:,:3]
mesh = ax[0].pcolormesh(depth_color[::-1], vmin=vmin, vmax=vmax, cmap=cmap_name)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=10*size_mult)
cbar.ax.set_title("(meters)", fontsize=7*size_mult, y=1.015)
ax[0].set_title('Ground Truth\n (Depth)', fontsize=12*size_mult)
ax[0].set_xticks([])
ax[0].set_yticks([])


titles = ['Proposed', 'Maximum Periodogram']
for i in range(2):
    idx = i
    file = os.path.join(results_dir, results_filekey[idx][0])
    key = 'estimator_' + results_filekey[idx][1]
    data = np.load(file)
    d_hats = data[key]

    depth = np.zeros_like(true_depth)
    depth[true_depth>0] = d_hats[:,0]
    cmap = plt.get_cmap(cmap_name)
    
    depth_color = np.ones((*depth.shape, 3), dtype=depth.dtype)
    depth_color[true_depth>0] = cmap((d_hats[:,0]-vmin)/(vmax-vmin))[:,:3]
    

    mesh = ax[i+1].pcolormesh(depth_color[::-1], vmin=vmin, vmax=vmax, cmap=cmap_name)
    divider = make_axes_locatable(ax[i+1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=10*size_mult)
    cbar.ax.set_title("(meters)", fontsize=7*size_mult, y=1.015)
    ax[i+1].set_title(titles[i]+'\n(Depth)', fontsize=12*size_mult)
    ax[i+1].set_xticks([])
    ax[i+1].set_yticks([])


    depth_abe = np.abs(true_depth - depth)
    if i==0:
        mesh = ax[i+3].pcolormesh(depth_abe[::-1], vmin=0, cmap='Reds')
        divider = make_axes_locatable(ax[i+3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar=fig.colorbar(mesh, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=10*size_mult)
        # ax[1, i+1].set_title(titles[i])
        ax[i+3].set_xticks([])
        ax[i+3].set_yticks([])
        ax[i+3].set_title(titles[i]+'\n(Absolute Error)', fontsize=12*size_mult)

    mesh = ax[i+4].pcolormesh(depth_abe[::-1], vmin=0, vmax=vmax, cmap='Reds')
    divider = make_axes_locatable(ax[i+4])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar=fig.colorbar(mesh, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=10*size_mult)
    cbar.ax.set_title("(meters)", fontsize=7*size_mult, y=1.015)
    # ax[1, i+1].set_title(titles[i])
    ax[i+4].set_xticks([])
    ax[i+4].set_yticks([])
    ax[i+4].set_title(titles[i]+'\n(Absolute Error)', fontsize=12*size_mult)
    
    print(data["param_name"])

plt.show()

