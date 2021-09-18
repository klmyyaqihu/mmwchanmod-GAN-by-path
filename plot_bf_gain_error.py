"""
plot_snr.py:  Plots the SNR distribution in a single cell environment.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse

import tensorflow.keras.backend as K

from tqdm import trange

# path = os.path.abspath('../..')
# if not path in sys.path:
#     sys.path.append(path)
    
from mmwchanmod.datasets.download import load_model 
from mmwchanmod.sim.antenna import Elem3GPP, ElemDipole
from mmwchanmod.learn.datastats import  data_to_mpchan 
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from mmwchanmod.sim.chanmod import dir_bf_gain_multi_sect
    
"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the SNR distribution')    
parser.add_argument('--model_dir',action='store',default= 'models/Beijing', 
    help='directory to store models')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='bf_gain_error.png', help='plot file name')        
parser.add_argument(\
    '--mod_name',action='store',\
    default='Beijing', help='model to load') 
    
args = parser.parse_args()
model_dir = args.model_dir
model_name = model_dir.split('/')[-1]
plot_dir = args.plot_dir
plot_fn = args.plot_fn
mod_name = args.mod_name

"""
load data
"""
# Load test data (.p format)
with open(model_dir+'/test_data.p', 'rb') as handle:
    test_data = pickle.load(handle)
with open(model_dir+'/cfg.p', 'rb') as handle:
    cfg = pickle.load(handle)

# Paramters
bw_f2 = 400e6   # f2 Bandwidth in Hz
bw_f1 = 20e6    # f1 Bandwidth in Hz
nf = 6  # Noise figure in dB
kT = -174   # Thermal noise in dBm/Hz
tx_pow = 23  # TX power in dBm
npts = 100    # number of points for each (x,z) bin
aer_height=30  # Height of the aerial cell in meterss
downtilt = 10  # downtilt in degrees

fc1 = 2.3e9 # freq 1
fc2 = 28e9  # carrier frequency in Hz freq 2
nant_gnb_fc1 = np.array([1,4])  # gNB array size Tx
nant_ue_fc1 = np.array([1,1])   # UE/UAV array size Rx
nant_gnb_fc2 = np.array([8,8])  # gNB array size Tx
nant_ue_fc2 = np.array([1,8])   # UE/UAV array size Rx
nsect = 3  # number of sectors for terrestrial gNBs 
    
# # Number of x and z bins
# nx = 40
# nz = 20

# # Range of x and z distances to test
# xlim = np.array([0,500])
# zlim = np.array([0,130])    

"""
Create the arrays
"""
# Terrestrial gNB.
# We downtilt the array and then replicate it over three sectors
elem_gnb_f1 = ElemDipole() # 2.3GHz
elem_gnb_f2 = Elem3GPP(thetabw=82, phibw=82) # 28GHz
arr_gnb0_f1 = URA(elem=elem_gnb_f1, nant=nant_gnb_fc1, fc=fc1)
arr_gnb0_f2 = URA(elem=elem_gnb_f2, nant=nant_gnb_fc2, fc=fc2)

arr_gnb_list_f1 = multi_sect_array(\
        arr_gnb0_f1, sect_type='azimuth', theta0=-downtilt, nsect=nsect)
arr_gnb_list_f2 = multi_sect_array(\
        arr_gnb0_f2, sect_type='azimuth', theta0=-downtilt, nsect=nsect)

# UE array.  Array is pointing down.
elem_ue_f1 = ElemDipole() # 2.3GHz
elem_ue_f2 = Elem3GPP(thetabw=82, phibw=82) # 28GHz
arr_ue0_f1 = URA(elem=elem_ue_f1, nant=nant_ue_fc1, fc=fc1)
arr_ue0_f2 = URA(elem=elem_ue_f2, nant=nant_ue_fc2, fc=fc2)
arr_ue_f1 = RotatedArray(arr_ue0_f1,theta0=90) # point up
arr_ue_f2 = RotatedArray(arr_ue0_f2,theta0=90)

"""
Load the pre-trained model
"""
    
# Construct and load the channel model object
mod_name = 'Beijing'
print('Loading pre-trained model %s' % mod_name)
K.clear_session()
chan_mod = load_model(model_name, cfg, 'local')
    

"""
Generate chan_list for test data
"""
# use the ray tracing data (real path data)
chan_list, ls = data_to_mpchan(test_data, cfg) 

dvec = test_data['dvec']
d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)
I = np.where((d3d>=100)& (d3d<=200))[0].astype(int) # only plot distance in (100,200) m

"""
Plot SNR 2.3 vs 28
"""
snr_plot = np.zeros((len(I), 2))
bf_gain_error_ls = []
for i, itest in enumerate(I):
    is_valid, bf_gain_error = dir_bf_gain_multi_sect(\
                    arr_gnb_list_f1, arr_gnb_list_f2, \
                    [arr_ue_f1], [arr_ue_f2], chan_list[itest])
    if is_valid:
        bf_gain_error_ls.append(bf_gain_error[0][0])


plt.plot(np.sort(bf_gain_error_ls), np.arange(len(bf_gain_error_ls))/len(bf_gain_error_ls)) 
plt.xlabel('Beamforming Gain Error (dB)', fontsize=14)
plt.ylabel('CDF', fontsize=14)
plt.grid()
# Print plot

# Construct and load the channel model object
mod_name = 'Beijing'
print('Loading pre-trained model %s' % mod_name)
K.clear_session()
chan_mod = load_model(model_name, cfg, 'local')
    
"""
Generate chan_list for test data
"""
fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
for ifreq in range(cfg.nfreq):
    fspl_ls[ifreq,:] = test_data['fspl' + str(ifreq+1)]

# Generate chan by the trained model (input is 'dvec')
chan_list, ls = chan_mod.sample_path(test_data['dvec'], \
                                    fspl_ls, test_data['rx_type'])
dvec = test_data['dvec']
d3d = np.maximum(np.sqrt(np.sum(dvec**2, axis=1)), 1)
I = np.where((d3d>=100)& (d3d<=200))[0].astype(int) # only plot distance in (100,200) m

"""
Plot SNR 2.3 vs 28
"""
snr_plot = np.zeros((len(I), 2))
bf_gain_error_ls = []
for i, itest in enumerate(I):
    is_valid, bf_gain_error = dir_bf_gain_multi_sect(\
                    arr_gnb_list_f1, arr_gnb_list_f2, \
                    [arr_ue_f1], [arr_ue_f2], chan_list[itest])
    # Compute the effective SNR
    if is_valid:
        bf_gain_error_ls.append(bf_gain_error[0][0])

plt.plot(np.sort(bf_gain_error_ls), np.arange(len(bf_gain_error_ls))/len(bf_gain_error_ls))
plt.legend(['Real Data', 'Train Model'])
plt.xlim(-1,25)

if 1:
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print('Created directory %s' % plot_dir)
    plot_path = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_path)
    print('Figure saved to %s' % plot_path)








# Old William code

# """
# Main simulation loop
# """      
# snr_med = np.zeros((nz,nx,nplot))

    
# for iplot, rx_type0 in enumerate(rx_types):

    
#     # Print cell type
#     print('')
#     print('Simulating RX type: %s' % rx_type0)
    
#     # Set the limits and x and z values to test
#     dx = np.linspace(xlim[0],xlim[1],nx)        
#     dz = np.linspace(zlim[0],zlim[1],nz)
#     if rx_type0 == 'Aerial':
#         dz = dz - aer_height
    
    
#     # Convert to meshgrid
#     dxmat, dzmat = np.meshgrid(dx,dz)
    
#     # Create the condition 
#     ns = nx*nz
#     phi = np.random.uniform(0,2*np.pi,ns)
#     dx = dxmat.ravel()
#     dz = dzmat.ravel()
#     dvec = np.column_stack((dx*np.cos(phi), dx*np.sin(phi), dz))
#     rx_type_vec = np.tile(iplot, (ns,))
        
        
#     # Loop over multiple trials
#     snr = np.zeros((nz,nx,npts))

#     # Generate frequencies list
#     fspl_ls = np.zeros((cfg.nfreq, test_data['dvec'].shape[0]))
#     for ifreq in range(cfg.nfreq):
#         fspl_ls[ifreq,:] = test_data['fspl' + str(ifreq+1)]
#     # Only keep dist bewteen 50 ~ 200m data



#     for i in range(npts):
#         # Generate random channels
#         chan_list, link_state = chan_mod.sample_path(dvec, rx_type_vec) 

#         # Compute the directional path loss of each link        
#         n = len(chan_list)
#         pl_gain = np.zeros(n)        
#         for j, c in enumerate(chan_list):            
#             if (rx_type0 == 'Aerial') and (nsect_a == 1):
#                 pl_gain[j] = dir_path_loss(arr_gnb_a, arr_ue, c)[0]
#             elif (rx_type0 == 'Aerial'):
#                 pl_gain[j] = dir_path_loss_multi_sect(\
#                     arr_gnb_list_a, [arr_ue], c)[0]
#             else:
#                 pl_gain[j] = dir_path_loss_multi_sect(\
#                     arr_gnb_list_t, [arr_ue], c)[0]
                                                   
#         # Compute the effective SNR
#         snri = tx_pow - pl_gain - kT - nf - 10*np.log10(bw)
    
#         # Create the data for the plot    
#         snri = snri.reshape((nz,nx))
#         snri = np.flipud(snri)
        
#         snr[:,:,i] = snri
     
#     # Get the median SNR
#     snr_med[:,:,iplot] = np.median(snr,axis=2) 
     
         
# # Plot the results
# for iplot, rx_type0 in enumerate(rx_types):
                    
#     plt.subplot(1,nplot,iplot+1)
#     plt.imshow(snr_med[:,:,iplot],\
#                extent=[np.min(xlim),np.max(xlim),np.min(zlim),np.max(zlim)],\
#                aspect='auto', vmin=-20, vmax=60)   
        
#     # Add horizontal line indicating location of aerial cell
#     if (rx_type0 == 'Aerial'):
#         plt.plot(xlim, np.array([1,1])*aer_height, 'r--')
        
#     if (iplot > 0):
#         plt.yticks([])
#     else:
#         plt.ylabel('Elevation (m)')
#     plt.xlabel('Horiz (m)')
#     plt.title(rx_types[iplot])
        

# # Add the colorbar
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, right=0.87, top=0.9)
# cax = plt.axes([0.92, 0.1, 0.05, 0.8])
# plt.colorbar(cax=cax)        
    
# if 1:
#     # Print plot
#     if not os.path.exists(plot_dir):
#         os.mkdir(plot_dir)
#         print('Created directory %s' % plot_dir)
#     plot_path = os.path.join(plot_dir, plot_fn)
#     plt.savefig(plot_path)
#     print('Figure saved to %s' % plot_path)
#     plt.savefig('snr_dist.png', bbox_inches='tight')
            
    


