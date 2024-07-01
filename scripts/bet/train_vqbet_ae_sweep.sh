

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_tune_1 \
    variant_name=block_1_beta_0.5 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=1 \
    algo.beta=0.5 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_tune_1/block_1_beta_0.5/stage_0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_tune_1 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=5 \
    algo.beta=0.5 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_tune_1/block_5_beta_0.5/stage_0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_tune_1 \
    variant_name=block_10_beta_0.5 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=10 \
    algo.beta=0.5 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_tune_1/block_10_beta_0.5/stage_0







sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_tune_1 \
    variant_name=block_1_beta_0.1 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=1 \
    algo.beta=0.1 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_tune_1/block_1_beta_0.1/stage_0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_tune_1 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=5 \
    algo.beta=0.1 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_tune_1/block_5_beta_0.1/stage_0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_tune_1 \
    variant_name=block_10_beta_0.1 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=10 \
    algo.beta=0.1 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_tune_1/block_10_beta_0.1/stage_0





