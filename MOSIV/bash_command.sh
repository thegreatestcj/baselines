python train_dynamic_MO.py -c config/genesismo/03.json -s data/03 -m output/03 --reg_scale --reg_alpha --use_wandb

python train_dynamic_MO_LQR.py -c config/genesismo/03.json -s data/03 -m selected_gic_pc_dataset_45_new/03 --reg_scale --reg_alpha --use_wandb


python train_gs_fixed_pcd.py -c config/genesismo/03.json -s data/03 -m selected_gic_pc_dataset_45_new/03
