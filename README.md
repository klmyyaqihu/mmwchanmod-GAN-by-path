# Millimeter Wave Channel Modeling via Generative Adversarial Network


## data
'dict.p' (from Beijing map) stores all raw data required for GAN
'data.csv' can show the insights of 'dict.p'

## training model from scratch	
Go to the current folder and run(e.g.):
```
python train_mod_interactive.py --model_dir models/model_name --nepochs_path 100 
```

'train_mod_interactive.py' has commands to change the number of epochs and model parameters.

## the folder 'models' stores a example trained model
'Conditional_WGAN_GP'

## the folder 'plots' stores path-loss and angle plots for the example model
there are plots for 10000, 15000, and 20000 epochs

## plot the path-loss cdf
Go to the current folder and run(e.g.):
```
python plot_path_loss_cdf.py  --model_dir models/model_name --plot_fn plot_name.png
```

## plot the angle distribution
Go to the current folder and run(e.g.):
```
python plot_angle_dist.py  --model_dir models/model_name --plot_fn plot_name.png
```

## the folder 'nnArch' stores the visualization of current network structures
-- dsc.png is the structure of discriminator
-- gen.png is the structure of generator

## To modify the code for the network, please go to "mmwchanmod/learn/models.py". In "models.py": 
1. the class 'CondGAN' describes the structure of GAN
2. 'fit_path_mod' in class 'ChanMod' is where we train GAN


## plot the SNR
-- model: python plot_snr_cmp_model.py --model_dir models/WGAN-GP --plot_fn xxx.png
-- data: python plot_snr_cmp_data.py --model_dir models/WGAN-GP --plot_fn xxx.png