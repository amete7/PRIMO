
python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=act \
    training.use_amp=false \
    train_dataloader.num_workers=6 \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    training.n_epochs=200 \
    seed=0









sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=act \
    exp_name=act_d256_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    training.n_epochs=200 \
    training.resume=false \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=act \
    exp_name=act_d512_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    training.n_epochs=200 \
    training.resume=false \
    seed=0


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=act \
    exp_name=act_d256_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    training.use_amp=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    training.n_epochs=200 \
    training.resume=false \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=act \
    exp_name=act_d512_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    training.use_amp=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.embed_dim=512 \
    training.n_epochs=200 \
    training.resume=false \
    seed=0

# KL weight 100


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=act \
    exp_name=act_d256_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    training.n_epochs=200 \
    training.resume=false \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=act \
    exp_name=act_d512_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    algo.kl_weight=100 \
    training.n_epochs=200 \
    training.resume=false \
    seed=0


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=act \
    exp_name=act_d256_kl100_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    training.use_amp=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    training.n_epochs=200 \
    training.resume=false \
    seed=0

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=act \
    exp_name=act_d512_kl100_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    training.use_amp=false \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.embed_dim=512 \
    algo.kl_weight=100 \
    training.n_epochs=200 \
    training.resume=false \
    seed=0


# # for debugging
# python train.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     algo=act \
#     exp_name=act_debug_noamp \
#     variant_name=block_16 \
#     training.use_tqdm=true \
#     training.use_amp=false \
#     training.save_all_checkpoints=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=true \
#     algo.skill_block_size=16 \
#     algo.frame_stack=1 \
#     training.n_epochs=200 \
#     training.resume=false \
#     logging.mode=disabled \
#     training.do_profile=True \
#     task.env_runner.debug=true \
#     seed=0