"""
plot_dly_rms:  Plots the CDF of the path loss on the test data,
and compares that to the randomly generated path loss from the trained model.

To compare the test data and model in London-Tokyo:
    
    python plot_path_loss_cdf.py --model_city LonTok --plot_fn pl_cdf_lt.png

To compare the test data in London-Tokyo and all models:
    
    python plot_path_loss_cdf.py --model_city "LonTok Moscow Beijing Boston"
        --plot_fn pl_cdf_all.png    
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K


# path = os.path.abspath('../..')
# if not path in sys.path:
#     sys.path.append(path)
    
from mmwchanmod.common.constants import LinkState
from mmwchanmod.learn.models import ChanMod
from mmwchanmod.datasets.download import get_dataset, load_model
from mmwchanmod.learn.datastats import  data_to_mpchan 
    

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the rms excess delay versus distance')
parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
    help='directory to store models')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='rms_dly_dist_correclation.png', help='plot file name')        
parser.add_argument(\
    '--ds_city',action='store',\
    default='Beijing', help='data set to load')    
parser.add_argument(\
    '--model_city',action='store',\
    default='Beijing', help='cities for the models to test')    
        
args = parser.parse_args()
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
plot_dir = args.plot_dir
ds_city = args.ds_city
model_city = args.model_city
plot_fn = args.plot_fn

"""
load data
"""
# Load test data (.p format)
with open(model_dir+'/test_data.p', 'rb') as handle:
    test_data = pickle.load(handle)
with open(model_dir+'/cfg.p', 'rb') as handle:
    cfg = pickle.load(handle)

# Cities to test are the city for the followed by all the models
city_test = [ds_city] + model_city.split() 
    
use_true_ls = False
    
     
"""
Find the RMS delays
"""
rms_dly_plot = []
rms_dly_plot_2 = []
ls_plot = []
leg_str = []

ntest = len(city_test)
fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
for ifreq in range(cfg.nfreq):
    fspl_ls[ifreq,:] = test_data['fspl' + str(ifreq+1)]
for i, city in enumerate(city_test):
    
    if (i == 0):
        """
        For first city, use the city data
        """       
        # Convert data to channel list
        chan_list, ls = data_to_mpchan(test_data, cfg)
        # leg_str.append(city + ' data')
        
        
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
            
        # leg_str.append(city + ' model')            
            
        
    # Compute the rms delay spread for each link    
    n = len(chan_list)
    rms_dly = np.zeros(n)
    rms_dly_2 = np.zeros(n)
    dvec = test_data['dvec']
    rms_dist = np.sqrt(dvec[:,0]**2 + dvec[:,1]**2+dvec[:,2]**2)
    for i, chan in enumerate(chan_list):
        if chan.link_state != LinkState.no_link:
            rms_dly[i], rms_dly_2[i] = chan.rms_dly(fspl_ls[:,i])
    
    # Save the results    
    ls_plot.append(ls)
    rms_dly_plot.append(rms_dly)
    rms_dly_plot_2.append(rms_dly_2)
            
"""
Create the plot
"""
ntypes = len(cfg.rx_types)
nplot = len(rms_dly_plot)
# plt.figure(figsize=(10,5))
# fig, ax = plt.subplots(1, ntypes)
d_min = 0
d_max = 1300
d_step = 10
n_step = (d_max-d_min)//d_step
d_plot = np.linspace(d_min,d_max,n_step)
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
ax_ls = [ax1, ax2]
c = ['b','r']
corr = []
for i, rx_type in enumerate(cfg.rx_types):
    
    # Plot color    
    for iplot in range(nplot):

        # Find the links that match the type and are not in outage
        I = np.where((test_data['rx_type']==i)\
                    & (ls_plot[iplot] != LinkState.no_link) \
                    & (rms_dly_plot[iplot]>1e-9))[0]
            
        # Select color and fmt
        if (iplot == 0):
            fmt = '-'
            color = [0,0,1]
        else:
            fmt = '--'
            t = (iplot-1)/(nplot-1)
            color = [0,t,1-t]

        # Plot the omni-directional path loss                 
        # ni = len(I)
        # p = np.arange(ni)/ni  
        # dlyi = np.sort(rms_dly_plot[iplot][I])*1e9
        dist = rms_dist[I]
        dly = rms_dly_plot[iplot][I]
        dly_2 = rms_dly_plot_2[iplot][I]
        I_sorted = np.argsort(dist)
        sorted_dist = dist[I_sorted]
        sorted_dly = dly[I_sorted]
        sorted_dly_2 = dly_2[I_sorted]
        # print(sorted_dist)
        dly_ls = []
        dly_second_ls = []
        for i_step in range(n_step):
            idx = np.where(((d_min+(i_step)*d_step)<=dist) & (dist<(d_min+(i_step+1)*d_step)))[0]
            dly_mean = np.sum(dly[idx])/len(idx)
            dly2_mean = np.sum(dly_2[idx])/len(idx)
            dly_ls.append(dly_mean*1e9)
            dly_second_ls.append(dly2_mean*1e9)
        
        # plt.scatter(dly_ls,dly_second_ls)
        data = pd.DataFrame()
        data['2.3 GHz RMS delay (ns)'] = dly_ls
        data['28 GHz RMS delay (ns)'] = dly_second_ls
        # sns_plot = sns.scatterplot(x="2.3", y="28", data=data)
        sns.regplot(x="2.3 GHz RMS delay (ns)", y="28 GHz RMS delay (ns)", data=data,  ax=ax_ls[iplot], color = c[iplot])

        corr_, _ = pearsonr(dly_ls, dly_second_ls)
        print('Correlation: %.3f' % corr_) 
        corr.append(corr_)

    # plt.set_title(rx_type)
    # g.set_ylabels('28 GHz RMS delayRMS delay (ns)')
    # g.set_xlabels('2.3 GHz RMS delay (ns)')
    ax1.set_title('Data r: %.3f' % corr[0])
    ax2.set_title('Model r: %.3f' % corr[1])
    ax1.grid()
    ax2.grid()
    fig.suptitle("RMS Delay (Avg. every 10m) Pearson Correlation")
    ax1.set_xlim([10, 320])
    ax2.set_xlim([10, 320])


# leg_str = ['2.3G Data',  '28G Data', '2.3G Model','28G Model']
# plt.legend(leg_str, borderaxespad=0.1, loc='upper right',\
#            bbox_to_anchor=(0, 0.15, 1, 0.85))
# plt.title('RMS Delay (Avg. 10m, Data) Spearmans correlation: %.3f' % corr)
# g.set_titles('RMS Delay (Avg. 10m, Model) Pearson Correlation: %.3f' % corr) 
# plt.subplots_adjust(right=0.85)
    
# Print plot
if 1:
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print('Created directory %s' % plot_dir)
    plot_path = os.path.join(plot_dir, plot_fn)
    fig.savefig(plot_path)
    print('Figure saved to %s' % plot_path)

    


