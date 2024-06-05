python train.py --config-name=train_autoencoder.yaml logging.mode=disabled train_dataloader.persistent_workers=false


python train.py --config-name=train_prior.yaml logging.mode=disabled











# Batch runs


sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    exp_name=implementing_3 \
    make_unique_experiment_dir=false \
    train_dataloader.persistent_workers=true



    

