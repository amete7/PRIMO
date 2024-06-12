# python evaluate.py \
#     exp_name=eval_tune_1_prior_2 \
#     variant_name=block_64_ds_2 \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=8 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_32_ds_8/0/stage_1/

# python evaluate.py \
#     exp_name=eval_tune_1_prior_2 \
#     variant_name=block_16_ds_2 \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2 \
#     training.use_tqdm=false \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_16_ds_2/0/stage_1



sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    exp_name=eval_2_tune_1_prior_2 \
    variant_name=block_16_ds_2 \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_16_ds_2/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    exp_name=eval_2_tune_1_prior_2 \
    variant_name=block_16_ds_4 \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_16_ds_4/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    exp_name=eval_2_tune_1_prior_2 \
    variant_name=block_16_ds_8 \
    algo.skill_block_size=16 \
    algo.downsample_factor=8 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_16_ds_8/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    exp_name=eval_2_tune_1_prior_2 \
    variant_name=block_32_ds_2 \
    algo.skill_block_size=32 \
    algo.downsample_factor=2 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_32_ds_2/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    exp_name=eval_2_tune_1_prior_2 \
    variant_name=block_32_ds_4 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_32_ds_4/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    exp_name=eval_2_tune_1_prior_2 \
    variant_name=block_32_ds_8 \
    algo.skill_block_size=32 \
    algo.downsample_factor=8 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_32_ds_8/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    exp_name=eval_2_tune_1_prior_2 \
    variant_name=block_64_ds_2 \
    algo.skill_block_size=64 \
    algo.downsample_factor=2 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_64_ds_2/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    exp_name=eval_2_tune_1_prior_2 \
    variant_name=block_64_ds_4 \
    algo.skill_block_size=64 \
    algo.downsample_factor=4 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_64_ds_4/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    exp_name=eval_2_tune_1_prior_2 \
    variant_name=block_64_ds_8 \
    algo.skill_block_size=64 \
    algo.downsample_factor=8 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45/quest/tune_1_prior_2/block_64_ds_8/0/stage_1











