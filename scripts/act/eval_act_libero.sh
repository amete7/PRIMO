seeds=(0 1 2 3)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python evaluate.py \
        task=libero_90 \
        algo=act_policy \
        exp_name=act_d256 \
        variant_name=block_16 \
        stage=1 \
        training.use_tqdm=false \
        algo.skill_block_size=16 \
        algo.embed_dim=256 \
        seed=$seed
done