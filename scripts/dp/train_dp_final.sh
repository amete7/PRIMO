

python fix.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=diffusion_policy \
    exp_name=dp_final \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    training.n_epochs=200 \
    training.resume=true \
    seed=0



python fix.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=diffusion_policy \
    exp_name=dp_final \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    training.n_epochs=200 \
    training.resume=true \
    seed=1


python fix.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=diffusion_policy \
    exp_name=dp_final \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    training.n_epochs=200 \
    training.resume=true \
    seed=2

