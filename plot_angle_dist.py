"""
plot_angle_dist.py:  Plots the angular distribution

For all the NLOS paths, the program:
* Computes the  AoA and AoD relative to the LOS path
* Plots the empirical distribution of the relative angles as 
  a function of the distance
* Generates random angles with the same conditions as the model,
  and plots the relative angle as a function of the distance
  for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import argparse
import os
import sys
import pickle

# path = os.path.abspath('../..')
# if not path in sys.path:
#     sys.path.append(path)
    
from mmwchanmod.common.constants import  AngleFormat
from mmwchanmod.learn.models import ChanMod
from mmwchanmod.datasets.download import load_model

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the angle vs. distance')
parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
    help='directory to store models')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='angle_dist.png', help='plot file name')        
parser.add_argument(\
    '--ds_name',action='store',\
    default='Beijing', help='data set to load')    
    
args = parser.parse_args()
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
plot_dir = args.plot_dir
plot_fn = args.plot_fn
ds_name = args.ds_name


def plot_ang_dist(ax,chan_mod,dvec,fspl_ls,nlos_ang,nlos_pl_ls,iang,pl_tol=60, dmax=500):
    """
    Plots the conditional distribution of the relative angle.
    
    Parameters
    ----------
    ax : pyplot axis
        Axis to plot on
    chan_mod : ChanMod structure
        Channel model.
    dvec : (nlink,ndim) array
            vector from cell to UAV
    fspl1 & fspl2: (nlink,) array
    nlos_ang : (nlink,npaths_max,nangle) array
            Angles of each path in each link.  
            The angles are in degrees
    nlos_pl : (nlink,npaths_max) array 
            Path losses of each path in each link.
            A value of pl_max indicates no path
    iang: integer from 0 to DataFormat.nangle-1
        Index of the angle to be plotted
    np_plot:  integer
        Number of paths whose angles are to be plotted
    """
    # Get the distances
    dist = np.sqrt(np.sum(dvec**2,axis=1))
    dist_plot = np.tile(dist[:,None],(1,chan_mod.npaths_max))
    dist_plot = dist_plot.ravel()
    
    # Transform the angles.  The transformations compute the
    # relative angles and scales them by 180
    ang_tr = chan_mod.transform_ang(dvec, fspl_ls, nlos_ang, nlos_pl_ls)
    ang_rel = ang_tr[:,iang*chan_mod.npaths_max:(iang+1)*chan_mod.npaths_max]*180
    ang_rel = ang_rel.ravel()
    
    # Find valid paths based on the first frequency
    pl_tgt = fspl_ls[0,:]+pl_tol
    Ivalid = (nlos_pl_ls[0,:,:] < pl_tgt[:,None]-0.1)
    Ivalid = np.where(Ivalid.ravel())[0]

    # Get the valid distances and relative angles
    ang_rel = ang_rel[Ivalid]
    dist_plot = dist_plot[Ivalid]      
    
    # Set the angle and distance range for the historgram
    drange = [0,dmax]
    if iang==AngleFormat.aoa_phi_ind or iang==AngleFormat.aod_phi_ind:
        ang_range = [-180,180]
    elif iang==AngleFormat.aoa_theta_ind or iang==AngleFormat.aod_theta_ind:
        ang_range = [-90,90]
    else:
        raise ValueError('Invalid angle index')
    
    # Compute the emperical conditional probability
    H0, dedges, ang_edges = np.histogram2d(dist_plot,ang_rel,bins=[10,40],\
                                           range=[drange,ang_range])       
    Hsum = np.sum(H0,axis=1)
    H0 = H0 / Hsum[:,None]
    
    # Plot the log probability.
    # We plot the log proability since the probability in linear
    # scale is difficult to view
    log_prob = np.log10(np.maximum(0.01,H0.T))
    im = ax.imshow(log_prob, extent=[np.min(dedges),np.max(dedges),\
               np.min(ang_edges),np.max(ang_edges)], aspect='auto')   
    return im


"""
Load the true data
"""
# Load test data (.p format)
with open(model_dir+'/test_data.p', 'rb') as handle:
    real_data = pickle.load(handle)
with open(model_dir+'/cfg.p', 'rb') as handle:
    cfg = pickle.load(handle)

"""
Run the model
"""
    
# Construct the channel model object
K.clear_session()
chan_mod = load_model(model_name, cfg, 'local')

fspl_ls = np.zeros((cfg.nfreq, real_data['dvec'].shape[0]))
for ifreq in range(cfg.nfreq):
    fspl_ls[ifreq,:] = real_data['fspl' + str(ifreq+1)]

# Generate samples from the path
sim_data = chan_mod.sample_path(\
    real_data['dvec'], fspl_ls, real_data['rx_type'], real_data['link_state'], return_dict=True)
    
chan_mod0 = ChanMod(cfg=cfg)
    
"""
Plot the angular distributions
"""    
plt.rcParams.update({'font.size': 12})

ang_str = ['AoD Az', 'AoD El', 'AoA Az', 'AoA El']
    
fig, ax = plt.subplots(AngleFormat.nangle, 2, figsize=(5,10))
for iang in range(AngleFormat.nangle):
    
    for j in range(2):
        if j == 0:
            data = real_data
        else:
            data = sim_data
                        
        axi = ax[iang,j]

        fspl_ls = np.zeros((cfg.nfreq, data['dvec'].shape[0]))
        for ifreq in range(cfg.nfreq):
            fspl_ls[ifreq,:] = data['fspl' + str(ifreq+1)]
        nlos_pl_ls = np.zeros((cfg.nfreq, data['nlos_pl'].shape[0], data['nlos_pl'].shape[1]))
        for ifreq in range(cfg.nfreq):
            if ifreq == 0:
                nlos_pl_ls[ifreq,:,:] = data['nlos_pl']
            else:
                nlos_pl_ls[ifreq,:,:] = data['nlos_pl' + str(ifreq+1)]
        im = plot_ang_dist(axi,chan_mod0,data['dvec'], fspl_ls, data['nlos_ang'],\
                      nlos_pl_ls,iang, dmax=800)
            
        if iang < 3:
            axi.set_xticks([])
        else:
            axi.set_xlabel('Dist (m)')
        if j == 1:
            axi.set_yticks([])
            title_str = ang_str[iang] + ' Model'   
        else:
            title_str = ang_str[iang] + ' Data'   
        axi.set_title(title_str)
fig.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

if 1:     
    # Save the figure
    plot_path = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_path)
