# python evaluate.py \
#     task=metaworld_ml45_prise \
#     exp_name=eval_3_tune_2_prior \
#     variant_name=block_16_ds_2_ah_2 \
#     algo.action_horizon=4 \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2 \
#     training.use_tqdm=true \
#     rollout.rollouts_per_env=1 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_2/0/stage_1



# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise \
#     exp_name=eval_3_tune_2_prior \
#     variant_name=block_16_ds_2_ah_2 \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2 \
#     training.use_tqdm=false \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_2/0/stage_1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise \
#     exp_name=eval_3_tune_2_prior \
#     variant_name=block_16_ds_4_ah_2 \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=4 \
#     training.use_tqdm=false \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_4/0/stage_1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise \
#     exp_name=eval_3_tune_2_prior \
#     variant_name=block_16_ds_8_ah_2 \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=8 \
#     training.use_tqdm=false \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_8/0/stage_1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise \
#     exp_name=eval_3_tune_2_prior \
#     variant_name=block_32_ds_2_ah_2 \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=2 \
#     training.use_tqdm=false \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_2/0/stage_1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise \
#     exp_name=eval_3_tune_2_prior \
#     variant_name=block_32_ds_4_ah_2 \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=4 \
#     training.use_tqdm=false \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_4/0/stage_1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise \
#     exp_name=eval_3_tune_2_prior \
#     variant_name=block_32_ds_8_ah_2 \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=8 \
#     training.use_tqdm=false \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_8/0/stage_1































sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_16_ds_2_ah_8 \
    algo.action_horizon=8 \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_2/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_16_ds_4_ah_8 \
    algo.action_horizon=8 \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_4/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_16_ds_8_ah_8 \
    algo.action_horizon=8 \
    algo.skill_block_size=16 \
    algo.downsample_factor=8 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_8/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_32_ds_2_ah_8 \
    algo.action_horizon=8 \
    algo.skill_block_size=32 \
    algo.downsample_factor=2 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_2/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_32_ds_4_ah_8 \
    algo.action_horizon=8 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_4/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_32_ds_8_ah_8 \
    algo.action_horizon=8 \
    algo.skill_block_size=32 \
    algo.downsample_factor=8 \
    training.use_tqdm=false \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_8/0/stage_1

















sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_16_ds_2_ah_16 \
    algo.action_horizon=16 \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    training.use_tqdm=false \
    rollout.rollouts_per_env=20 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_2/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_16_ds_4_ah_16 \
    algo.action_horizon=16 \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    training.use_tqdm=false \
    rollout.rollouts_per_env=20 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_4/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_16_ds_8_ah_16 \
    algo.action_horizon=16 \
    algo.skill_block_size=16 \
    algo.downsample_factor=8 \
    training.use_tqdm=false \
    rollout.rollouts_per_env=20 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_8/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_32_ds_2_ah_16 \
    algo.action_horizon=16 \
    algo.skill_block_size=32 \
    algo.downsample_factor=2 \
    training.use_tqdm=false \
    rollout.rollouts_per_env=20 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_2/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_32_ds_4_ah_16 \
    algo.action_horizon=16 \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    training.use_tqdm=false \
    rollout.rollouts_per_env=20 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_4/0/stage_1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    exp_name=eval_3_tune_2_prior \
    variant_name=block_32_ds_8_ah_16 \
    algo.action_horizon=16 \
    algo.skill_block_size=32 \
    algo.downsample_factor=8 \
    training.use_tqdm=false \
    rollout.rollouts_per_env=20 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_8/0/stage_1












