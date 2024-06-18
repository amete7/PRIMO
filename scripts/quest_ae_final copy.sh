
sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    exp_name=quest_ae_final \
    variant_name=block_16_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    training.auto_continue=true \
    seed=0


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    exp_name=quest_ae_final \
    variant_name=block_16_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    training.auto_continue=true \
    seed=1


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    exp_name=quest_ae_final \
    variant_name=block_16_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    training.auto_continue=true \
    seed=2







sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    exp_name=quest_ae_final \
    variant_name=block_16_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    training.auto_continue=true \
    seed=0


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    exp_name=quest_ae_final \
    variant_name=block_16_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    training.auto_continue=true \
    seed=1


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    exp_name=quest_ae_final \
    variant_name=block_16_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    training.auto_continue=true \
    seed=2








