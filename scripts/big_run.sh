
sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=bs_256_block_16_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=256 \
    algo.lr=0.0001 \
    algo.skill_block_size=32 \
    algo.downsample_factor=2

sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=bs_1024_block_16_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=1024 \
    algo.lr=0.0004 \
    algo.skill_block_size=32 \
    algo.downsample_factor=2

sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=bs_4096_block_16_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=4096 \
    algo.lr=0.001 \
    algo.skill_block_size=32 \
    algo.downsample_factor=2




sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=bs_256_block_16_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=256 \
    algo.lr=0.0001 \
    algo.skill_block_size=16 \
    algo.downsample_factor=4

sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=bs_1024_block_16_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=1024 \
    algo.lr=0.0004 \
    algo.skill_block_size=16 \
    algo.downsample_factor=4

sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=bs_4096_block_16_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=4096 \
    algo.lr=0.001 \
    algo.skill_block_size=16 \
    algo.downsample_factor=4




sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=bs_256_block_16_ds_8 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=256 \
    algo.lr=0.0001 \
    algo.skill_block_size=16 \
    algo.downsample_factor=8

sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=bs_1024_block_16_ds_8 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=1024 \
    algo.lr=0.0004 \
    algo.skill_block_size=16 \
    algo.downsample_factor=8

sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=bs_4096_block_16_ds_8 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=4096 \
    algo.lr=0.001 \
    algo.skill_block_size=16 \
    algo.downsample_factor=8

