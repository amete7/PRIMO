python train.py --config-name=train_autoencoder.yaml \
    logging.mode=disabled \
    train_dataloader.persistent_workers=false \
    train_dataloader.batch_size=256


python train.py --config-name=train_prior.yaml logging.mode=disabled


python train.py --config-name=train_prior.yaml \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/implementing_3/stage_0/multitask_model_final.pth \
    train_dataloader.num_workers=6 \
    rollout.rollouts_per_env=3



sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=bs_256_block_32_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=256 \
    algo.lr=0.0001 \
    algo.skill_block_size=32 \
    algo.downsample_factor=2

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=bs_1024_block_32_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=1024 \
    algo.lr=0.0004 \
    algo.skill_block_size=32 \
    algo.downsample_factor=2

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=bs_4096_block_32_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=4096 \
    algo.lr=0.001 \
    algo.skill_block_size=32 \
    algo.downsample_factor=2




sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=bs_256_block_32_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=256 \
    algo.lr=0.0001 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=bs_1024_block_32_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=1024 \
    algo.lr=0.0004 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=bs_4096_block_32_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=4096 \
    algo.lr=0.001 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4




sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=bs_256_block_32_ds_8 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=256 \
    algo.lr=0.0001 \
    algo.skill_block_size=32 \
    algo.downsample_factor=8

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=bs_1024_block_32_ds_8 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=1024 \
    algo.lr=0.0004 \
    algo.skill_block_size=32 \
    algo.downsample_factor=8

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=bs_4096_block_32_ds_8 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    train_dataloader.batch_size=4096 \
    algo.lr=0.001 \
    algo.skill_block_size=32 \
    algo.downsample_factor=8












sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
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

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
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

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
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




sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
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

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
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

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
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




sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
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

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
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

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
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








# Batch runs

sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    exp_name=implementing_3 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    make_unique_experiment_dir=false \
    train_dataloader.persistent_workers=true



sbatch slurm/run_v100.sbatch python train.py --config-name=train_autoencoder.yaml \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    exp_name=implementing_3 \
    make_unique_experiment_dir=false \
    train_dataloader.persistent_workers=true

sbatch slurm/run_v100.sbatch python train.py --config-name=train_prior.yaml \
    exp_name=implementing_3 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.auto_continue=true \
    train_dataloader.persistent_workers=true \
    make_unique_experiment_dir=false




    

