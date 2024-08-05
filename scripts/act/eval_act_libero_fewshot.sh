seeds=(0 1 2 3 5 6 7 10 12 14)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python evaluate.py \
        task=libero_long_fewshot \
        algo=act_policy \
        exp_name=act_d256 \
        variant_name=block_16 \
        stage=2 \
        training.use_tqdm=false \
        algo.skill_block_size=16 \
        algo.embed_dim=256 \
        seed=$seed
done

# python evaluate.py \
#     task=libero_long_fewshot \
#     algo=act_policy \
#     exp_name=act_d256_latency \
#     variant_name=block_16 \
#     stage=2 \
#     training.use_tqdm=false \
#     algo.skill_block_size=16 \
#     algo.embed_dim=256 \
#     checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/libero/LIBERO_90/act_policy/act_d256/block_16/0/stage_1/multitask_model_final.pth \
#     seed=0