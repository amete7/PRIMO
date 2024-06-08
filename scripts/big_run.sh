

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_64_ds_2 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=64 \
    algo.downsample_factor=2 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_64_ds_2 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=64 \
    algo.downsample_factor=2 \
    seed=1


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_64_ds_4 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=64 \
    algo.downsample_factor=4 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_64_ds_4 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=64 \
    algo.downsample_factor=4 \
    seed=1

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_64_ds_8 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=64 \
    algo.downsample_factor=8 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_64_ds_8 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=64 \
    algo.downsample_factor=8 \
    seed=1













sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_32_ds_2_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=2 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_32_ds_2_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=2 \
    seed=1


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_32_ds_4_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_32_ds_4_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    seed=1

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_32_ds_8_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=8 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_32_ds_8_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=8 \
    seed=1














sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_16_ds_2_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_16_ds_2_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    seed=1

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_16_ds_4_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_16_ds_4_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    seed=1

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_16_ds_8_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=8 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=tune_1 \
    variant_name=block_16_ds_8_no_amp \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    training.resume=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=8 \
    seed=1







