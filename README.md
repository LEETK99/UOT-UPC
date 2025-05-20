# Unpaired Point Cloud Completion via Unbalanced Optimal Transport #

## Install ##

Before training, you should follow the 'Start-Requirements' and 'Install' from USSPA([USSPA](https://github.com/murcherful/USSPA))

For the 'Install', you should download code/util/{chamfer_distance, pointnet2_ops_lib} from USSPA([USSPA](https://github.com/murcherful/USSPA)) to ./util/

## Dataset ##

Get USSPA dataset from USSPA([USSPA](https://github.com/murcherful/USSPA))

Get PCN dataset from USSPA([USSPA](https://github.com/murcherful/USSPA))

## Training ##
```
cd network
```

#### UOT-UPC for USSPA dataset ####
```
python train.py --log_dir {log_dir} --TC T12 --class_name {'chair', 'table', 'trash_bin', 'tv_or_monitor', 'cabinet', 'bookshelf', 'sofa', 'lamp', 'bed', 'tub', 'all'} --tau1 0.044 --tau2 0.044 --phi1 softplus --phi2 softplus --cost_type infocd --betas_g 0.95 0.999 --betas_d 0.95 0.999 --dens_coord 10.5 
```
#### UOT-UPC for PCN dataset ####
```
python train.py --log_dir {log_dir} --TC T12 --class_name {'chair', 'table', 'cabinet', 'sofa', 'lamp'} --tau1 0.044 --tau2 0.044 --phi1 softplus --phi2 softplus --cost_type infocd --betas_g 0.95 0.999 --betas_d 0.95 0.999 --dens_coord 10.5 --pcn
```

## class imbalance  ##
#### UOT-UPC ####
```
python train.py --log_dir {log_dir} --TC T12 --class_name {'table, tv_or_monitor'} --target_ratio {0.3, 0.5, 0.7, 1} 1 --motivation --tau1 0.044 --tau2 0.044 --phi1 softplus --phi2 softplus --cost_type infocd --betas_g 0.95 0.999 --betas_d 0.95 0.999 --dens_coord 10.5 
```
#### OT-UPC ####
```
python train.py --log_dor {log_dir} --TC T12_OT --class_name {'table, tv_or_monitor'} --target_ratio {0.3, 0.5, 0.7, 1} 1 --motivation --tau1 0.044 --tau2 0.044 --clip 1 --r1_gamma 0.2 --r2_gamma 0.2 --phi1 linear --phi2 linear --lr_g 5e-5 --lr_d 1e-7 --dens_coord 0
```

## cost function ablation ##

#### cd^{l2}, l2, {cd^{lw}}_{fwd} setting ####
```
python train.py --log_dir {log_dir} --TC T12 --class_name {'chair', 'table', 'trash_bin', 'tv_or_monitor', 'cabinet', 'bookshelf', 'sofa', 'lamp', 'bed', 'tub', 'all'} --tau1 100 --tau2 100 --phi1 softplus --phi2 softplus --cost_type {cd, cd_fwd, l2} --betas_g 0.95 0.999 --betas_d 0.95 0.999 --dens_coord 10.5 
```
