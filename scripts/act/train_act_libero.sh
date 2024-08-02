seeds=(0 1 2 3 4)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
        task=libero_90 \
        algo=act_policy \
        exp_name=act_d256 \
        variant_name=block_16 \
        training.use_tqdm=false \
        training.use_amp=false \
        training.save_all_checkpoints=true \
        train_dataloader.persistent_workers=true \
        train_dataloader.num_workers=6 \
        make_unique_experiment_dir=false \
        algo.skill_block_size=16 \
        algo.embed_dim=256 \
        training.n_epochs=100 \
        training.resume=false \
        seed=$seed
done