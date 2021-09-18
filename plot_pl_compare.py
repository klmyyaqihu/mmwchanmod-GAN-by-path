"""
plot_path_loss_cdf2:  Plots the CDF of the path loss on the test data,
and compares that to the randomly generated path loss from the trained model.
"""
import os
import pickle
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

# path = os.path.abspath('../..')
# if not path in sys.path:
#     sys.path.append(path)

from mmwchanmod.common.constants import LinkState
from mmwchanmod.datasets.download import load_model
from mmwchanmod.learn.datastats import  data_to_mpchan 
from mmwchanmod.common.constants import DataConfig
from mmwchanmod.learn.models import ChanMod

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the omni directional path loss CDF')
parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
    help='directory to store models')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='pl_compare_two_freq.png', help='plot file name')        
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

city_test = [ds_city] + model_city.split()
use_true_ls = False

"""
Find the path loss CDFs
"""
pl_omni_plot = []
pl2_omni_plot = []
ls_plot = []
leg_str = []

ntest = len(city_test)
for i, city in enumerate(city_test):
    
    if (i == 0):
        """
        For first city, use the city data
        """
        # Convert data to channel list
        chan_list, ls = data_to_mpchan(test_data, cfg)
        
        leg_str.append(city + ' data')
        
        
    else:
        """
        For subsequent cities, generate data from model
        """
        # Construct the channel model object
        K.clear_session()
        chan_mod = load_model(model_name,cfg, 'local')
        
        mod_name = 'Beijing'
        # Load the configuration and link classifier model
        print('Simulating model %s'%mod_name)        
        
        # Generate samples from the path
        if use_true_ls:
            ls = test_data['link_state']
        else:
            ls = None
        fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
        for ifreq in range(cfg.nfreq):
            fspl_ls[ifreq,:] = test_data['fspl' + str(ifreq+1)]
        chan_list, ls = chan_mod.sample_path(test_data['dvec'], fspl_ls, test_data['rx_type'], ls)
            
        leg_str.append(city + ' model') 

        
    # Compute the omni-directional path loss for each link    
    n = len(chan_list)
    pl_omni = np.zeros(n)
    pl2_omni = np.zeros(n)
    for i, chan in enumerate(chan_list):
        if chan.link_state != LinkState.no_link:
            pl_omni[i], pl2_omni[i]= chan.comp_omni_path_loss()

    # Save the results    
    ls_plot.append(ls)
    pl_omni_plot.append(pl_omni)
    pl2_omni_plot.append(pl2_omni)

"""
Create the plot
"""
ntypes = len(cfg.rx_types)
nplot = len(pl_omni_plot)
# plt.figure(figsize=(10,5))
# fig, ax = plt.subplots(1,2)
# markers = ['.','v']
colors = ['b','y']

for i, rx_type in enumerate(cfg.rx_types):
    
    # Plot color    
    for iplot in range(nplot):
        if iplot == 1:
        
            dvec = test_data['dvec']
            d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)

            # Find the links that match the type and are not in outage
            I = np.where((test_data['rx_type']==i) & (d3d>=100)& (d3d<=200)&\
                (ls_plot[iplot] != LinkState.no_link))[0]
            I_min = np.where(d3d == np.min(d3d[I]))[0][0]
            I_max = np.where(d3d == np.max(d3d[I]))[0][0]
                
            # Select color and fmt
            if (iplot == 0):
                fmt = '-'
                color = [0,0,1]
            else:
                fmt = '--'
                t = (iplot-1)/(nplot-1)
                color = [0,t,1-t]

            # Plot the omni-directional path loss                 
            ni = len(I)
            p = np.arange(ni)/ni            
            # plt.plot([pl_omni_plot[0][I_min],pl_omni_plot[0][I_max]], \
            #     [pl_omni_plot[0][I_min]+20*np.log10(28/2.3),pl_omni_plot[0][I_max]+20*np.log10(28/2.3)])

            plt.scatter(pl_omni_plot[iplot][I],pl2_omni_plot[iplot][I],s=9, c=colors[iplot])
        # plt.scatter(d3d[I], pl2_omni_plot[iplot][I], s=9)
              

plt.title('100-200(m) Model')   
plt.xlabel('2.3GHz Path loss(dB)')
plt.ylabel('28GHz Path loss(dB)')
plt.grid()

# plt.legend(['data_2.3GHz','data_28GHz', 'model_2.3GHz','model_28GHz'],loc='lower right',ncol=2)


# Print plot
if 1:
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print('Created directory %s' % plot_dir)
    plot_path = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_path)
    print('Figure saved to %s' % plot_path)