
sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    task=libero_90 \
    exp_name=quest_default \
    variant_name=block_32_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    training.resume=false \
    seed=0


# stage 0 long

# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
#     task=libero_long \
#     exp_name=quest_default \
#     variant_name=block_32_ds_4 \
#     training.use_tqdm=false \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=4 \
#     training.resume=false \
#     seed=0

# debugging

# python train.py --config-name=train_prior.yaml \
#     task=libero_long \
#     exp_name=lib_quest_debug \
#     variant_name=block_32_ds_4 \
#     training.use_tqdm=true \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=true \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=4 \
#     training.resume=false \
#     logging.mode=disabled \
#     training.auto_continue=true \
#     training.save_interval=1 \
#     rollout.interval=1 \
#     seed=0

# CUDA_LAUNCH_BLOCKING=1 python train.py --config-name=train_prior.yaml \
#     task=libero_long \
#     exp_name=lib_quest_debug \
#     variant_name=block_32_ds_4 \
#     training.use_tqdm=true \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     training.do_profile=true \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=true \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=4 \
#     training.resume=false \
#     logging.mode=disabled \
#     training.auto_continue=true \
#     training.save_interval=1 \
#     rollout.interval=1 \
#     seed=0

# python train.py --config-name=train_autoencoder.yaml \
#     task=libero_long \
#     exp_name=lib_quest_debug \
#     variant_name=block_32_ds_4 \
#     training.use_tqdm=true \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=4 \
#     training.resume=false \
#     logging.mode=disabled \
#     seed=0