
skill_block_size=16
downsample_factors=(2 4)
seeds=(0 1 2)

for downsample_factor in ${downsample_factors[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
            task=pusht \
            exp_name=quest_1024 \
            variant_name=block_${skill_block_size}_ds_${downsample_factor} \
            training.use_tqdm=false \
            training.save_all_checkpoints=true \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.skill_block_size=${skill_block_size} \
            algo.downsample_factor=$downsample_factor \
            training.resume=false \
            algo.policy.image_aug_factory=null \
            seed=$seed
    done
done