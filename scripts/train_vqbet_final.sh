# python train.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     algo=bet \
#     training.use_amp=true \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=5 \
#     algo.beta=0.5 \
#     seed=0 \
#     logging.mode=disabled






sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=5 \
    algo.beta=0.5 \
    seed=0


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=5 \
    algo.beta=0.5 \
    seed=1


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=5 \
    algo.beta=0.5 \
    seed=2









sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=5 \
    algo.beta=0.1 \
    seed=0


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=5 \
    algo.beta=0.1 \
    seed=1


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=5 \
    algo.beta=0.1 \
    seed=2






