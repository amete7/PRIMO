python train.py --config-name=train_autoencoder.yaml \
    algo=bet \
    task=metaworld_ml45_prise \
    logging.mode=disabled \
    train_dataloader.persistent_workers=false



python train.py --config-name=train_prior.yaml \
    algo=bet \
    task=metaworld_ml45_prise \
    logging.mode=disabled \
    train_dataloader.persistent_workers=false \
    checkpoint_path=/home/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/debug/run_000/multitask_model.pth





python train.py --config-name=train_fewshot.yaml \
    algo=quest \
    task=metaworld_ml45_prise_fewshot \
    logging.mode=disabled \
    train_dataloader.persistent_workers=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_2/0/stage_1/multitask_model_epoch_0090.pth \
    task.env_runner.debug=true





sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    exp_name=tune_2_prior \
    variant_name=block_16_ds_2 \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2/block_16_ds_2_no_amp/0/stage_0/




python train.py --config-name=train_prior.yaml logging.mode=disabled


python train.py --config-name=train_prior.yaml \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/implementing_3/stage_0/multitask_model_final.pth \
    train_dataloader.num_workers=6 \
    rollout.rollouts_per_env=3




sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
    training.use_tqdm=false \
    train_dataloader.num_workers=6

python train.py --config-name=train_autoencoder.yaml \
    task=metaworld_ml45_prise \
    algo.skill_block_size=64 \
    algo.downsample_factor=2 \
    logging.mode=disabled



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




    

