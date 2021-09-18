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
from mmwchanmod.common.constants import LinkState
from mmwchanmod.common.constants import  AngleFormat
from mmwchanmod.learn.models import ChanMod
from mmwchanmod.datasets.download import load_model
from mmwchanmod.learn.datastats import  data_to_mpchan 

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
    default='angle_compare.png', help='plot file name')        
parser.add_argument(\
    '--ds_name',action='store',\
    default='Beijing', help='data set to load')    
    
args = parser.parse_args()
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
plot_dir = args.plot_dir
plot_fn = args.plot_fn
ds_name = args.ds_name


def plot_ang_compare(ax, ang_f1, ang_f2):
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
    # # Get the distances
    # dist = np.sqrt(np.sum(dvec**2,axis=1))
    # # dist_plot = np.tile(dist[:,None],(1,chan_mod.npaths_max))
    # # dist_plot = dist_plot.ravel()
    # I_dist = np.where((dist>0)& (dist<=1300))[0]

    # # Transform the angles.  The transformations compute the
    # # relative angles and scales them by 180
    # ang_tr = chan_mod.transform_ang(dvec, fspl_ls, nlos_ang, nlos_pl_ls)
    # ang_rel = ang_tr[I_dist,iang*chan_mod.npaths_max:(iang+1)*chan_mod.npaths_max]*180
    
    # # Find strongest path
    # nlos_pl = nlos_pl_ls[0,I_dist]
    # nlos_pl2 = nlos_pl_ls[1,I_dist]
    # # I_stg_freq1 = []
    # # I_stg_freq2 = []
    # ang_plot = np.zeros([len(nlos_pl),2])
    # for i in range(len(nlos_pl)):
    #     # I_stg_freq1.append(np.where(nlos_pl[i]==np.max(nlos_pl[i]))[0][0])
    #     # I_stg_freq2.append(np.where(nlos_pl2[i]==np.max(nlos_pl2[i]))[0][0])
    #     I_stg_freq1 = np.where(nlos_pl[i]==np.max(nlos_pl[i]))[0][0]
    #     I_stg_freq2 = np.where(nlos_pl2[i]==np.max(nlos_pl2[i]))[0][0]
    #     ang_plot[i,0] = ang_rel[i][I_stg_freq1]
    #     ang_plot[i,1] = ang_rel[i][I_stg_freq2]

    im = ax.scatter(ang_f1,ang_f2, s = 1)
    
    return im

"""
Load the true data
"""
# Load test data (.p format)
with open(model_dir+'/test_data.p', 'rb') as handle:
    test_data = pickle.load(handle)
with open(model_dir+'/cfg.p', 'rb') as handle:
    cfg = pickle.load(handle)

"""
Run the model
"""
    
use_true_ls = False

fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
for ifreq in range(cfg.nfreq):
    fspl_ls[ifreq,:] = test_data['fspl' + str(ifreq+1)]
for i in range(2):
    
    if (i == 0):
        """
        For first city, use the city data
        """       
        # Convert data to channel list
        chan_list, ls = data_to_mpchan(test_data, cfg)
        # leg_str.append(city + ' data')

        n = len(chan_list)
        data_rms_aoa_phi_f1 = np.zeros(n)
        data_rms_aoa_phi_f2 = np.zeros(n)
        data_rms_aoa_theta_f1 = np.zeros(n)
        data_rms_aoa_theta_f2 = np.zeros(n)
        data_rms_aod_phi_f1 = np.zeros(n)
        data_rms_aod_phi_f2 = np.zeros(n)
        data_rms_aod_theta_f1 = np.zeros(n)
        data_rms_aod_theta_f2 = np.zeros(n)
        data_valid_link_idx = []
        # dvec = test_data['dvec']
        # rms_dist = np.sqrt(dvec[:,0]**2 + dvec[:,1]**2+dvec[:,2]**2)
        for i, chan in enumerate(chan_list):
            if chan.link_state != LinkState.no_link:

                no_path, data_rms_aoa_phi_f1[i], data_rms_aoa_phi_f2[i], \
                data_rms_aoa_theta_f1[i], data_rms_aoa_theta_f2[i], \
                data_rms_aod_phi_f1[i], data_rms_aod_phi_f2[i], \
                data_rms_aod_theta_f1[i], data_rms_aod_theta_f2[i] = chan.rms_angle(fspl_ls[:,i])
                if no_path == 0:
                    data_valid_link_idx.append(i)
        
        
    else:
        """
        For subsequent cities, generate data from model
        """
        
        # Construct the channel model object
        K.clear_session()
        chan_mod = load_model(model_name, cfg, 'local')
        mod_name = 'Beijing'
        # Load the configuration and link classifier model
        print('Simulating model %s'%mod_name)          
        
        # Generate samples from the path
        if use_true_ls:
            ls = test_data['link_state']
        else:
            ls = None

        chan_list, ls = chan_mod.sample_path(test_data['dvec'], fspl_ls, test_data['rx_type'], ls)
        n = len(chan_list)
        model_rms_aoa_phi_f1 = np.zeros(n)
        model_rms_aoa_phi_f2 = np.zeros(n)
        model_rms_aoa_theta_f1 = np.zeros(n)
        model_rms_aoa_theta_f2 = np.zeros(n)
        model_rms_aod_phi_f1 = np.zeros(n)
        model_rms_aod_phi_f2 = np.zeros(n)
        model_rms_aod_theta_f1 = np.zeros(n)
        model_rms_aod_theta_f2 = np.zeros(n)
        model_valid_link_idx = []
        for i, chan in enumerate(chan_list):
            if chan.link_state != LinkState.no_link:
                no_path, model_rms_aoa_phi_f1[i], model_rms_aoa_phi_f2[i], \
                model_rms_aoa_theta_f1[i], model_rms_aoa_theta_f2[i], \
                model_rms_aod_phi_f1[i], model_rms_aod_phi_f2[i], \
                model_rms_aod_theta_f1[i], model_rms_aod_theta_f2[i] = chan.rms_angle(fspl_ls[:,i])
                if no_path == 0:
                    model_valid_link_idx.append(i)

# fspl_ls = np.zeros((cfg.nfreq, real_data['dvec'].shape[0]))
# for ifreq in range(cfg.nfreq):
#     fspl_ls[ifreq,:] = real_data['fspl' + str(ifreq+1)]

# # Generate samples from the path
# sim_data = chan_mod.sample_path(\
#     real_data['dvec'], fspl_ls, real_data['rx_type'], real_data['link_state'], return_dict=True)
    
# chan_mod0 = ChanMod(cfg=cfg)
    
"""
Plot the angular distributions
"""    
plt.rcParams.update({'font.size': 12})

ang_str = ['AoD Az', 'AoD El', 'AoA Az', 'AoA El']
    

fig, ax = plt.subplots(AngleFormat.nangle, 2, figsize=(5,10))
# four angles
for iang in range(AngleFormat.nangle):
    # data, model
    for j in range(2):

        axi = ax[iang,j]
        if iang == 0:
            if j == 0:
                ang_f1 = data_rms_aoa_phi_f1[data_valid_link_idx]
                ang_f2 = data_rms_aoa_phi_f2[data_valid_link_idx]
            else:
                ang_f1 = model_rms_aoa_phi_f1[model_valid_link_idx]
                ang_f2 = model_rms_aoa_phi_f2[model_valid_link_idx]
        elif iang == 1:
            if j == 0:
                ang_f1 = data_rms_aoa_theta_f1[data_valid_link_idx]
                ang_f2 = data_rms_aoa_theta_f2[data_valid_link_idx]
            else:
                ang_f1 = model_rms_aoa_theta_f1[model_valid_link_idx]
                ang_f2 = model_rms_aoa_theta_f2[model_valid_link_idx]
        elif iang == 2:
            if j == 0:
                ang_f1 = data_rms_aod_phi_f1[data_valid_link_idx]
                ang_f2 = data_rms_aod_phi_f2[data_valid_link_idx]
            else:
                ang_f1 = model_rms_aod_phi_f1[model_valid_link_idx]
                ang_f2 = model_rms_aod_phi_f2[model_valid_link_idx]
        elif iang == 3:
            if j == 0:
                ang_f1 = data_rms_aod_theta_f1[data_valid_link_idx]
                ang_f2 = data_rms_aod_theta_f2[data_valid_link_idx]
            else:
                ang_f1 = model_rms_aod_theta_f1[model_valid_link_idx]
                ang_f2 = model_rms_aod_theta_f2[model_valid_link_idx]

        im = plot_ang_compare(axi, ang_f1, ang_f2)
            
        if iang == 3:
            axi.set_xlabel('2.3GHz (deg)')
        if j == 0:
            axi.set_ylabel('28GHz (deg)')
        if iang == 1 or iang == 3:
            axi.set_xlim([-1,30])
            axi.set_ylim([-1,30])
        if iang == 0 or iang == 2:
            axi.set_xlim([-3,180])
            axi.set_ylim([-3,180])

        if j == 1:
            axi.set_yticks([])
            title_str = ang_str[iang] + ' Model'   
        else:
            title_str = ang_str[iang] + ' Data'   
        axi.set_title(title_str)
fig.suptitle('RMS Angles 2.3GHz vs 28GHz')
fig.tight_layout()

# fig.subplots_adjust(right=0.8)

if 1:     
    # Save the figure
    plot_path = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_path)
    
    
    


    


