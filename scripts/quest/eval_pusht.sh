
skill_block_size=16
downsample_factors=(2 4)
seeds=(0 1 2)

for downsample_factor in ${downsample_factors[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            task=pusht \
            exp_name=quest_1024 \
            variant_name=block_${skill_block_size}_ds_${downsample_factor} \
            stage=1 \
            training.use_tqdm=false \
            algo.skill_block_size=${skill_block_size} \
            algo.downsample_factor=$downsample_factor \
            rollout.rollouts_per_env=50 \
            rollout.n_video=50 \
            seed=$seed
    done
done

python evaluate.py \
    task=pusht \
    exp_name=quest_1024 \
    variant_name=block_16_ds_2 \
    stage=1 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    rollout.rollouts_per_env=50 \
    seed=0