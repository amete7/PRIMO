
blocks=(4 8 16)
seeds=(0 1)

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_mt50 \
            algo=act \
            exp_name=act_baseline \
            variant_name=reimp_2_block_${block} \
            training.use_tqdm=false \
            training.use_amp=false \
            training.save_all_checkpoints=true \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.skill_block_size=${block} \
            algo.embed_dim=256 \
            training.n_epochs=2000 \
            training.resume=false \
            seed=${seed}
    done
done