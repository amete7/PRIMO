# python evaluate.py --config-name=evaluate.yaml \
#     task=metaworld_ml45_prise \
#     algo=quest \
#     exp_name=final_eval_2_test \
#     variant_name=quest_block_16_ds_2 \
#     training.use_tqdm=true \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2 \
#     seed=0 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_ae_final/block_16_ds_2/0/stage_1 \
#     rollout.rollouts_per_env=1


# QueST

sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=quest \
    exp_name=final_eval_2 \
    variant_name=quest_block_16_ds_2 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_ae_final/block_16_ds_2/0/stage_1


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=quest \
    exp_name=final_eval_2 \
    variant_name=quest_block_16_ds_2 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    seed=1 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_ae_final/block_16_ds_2/1/stage_1


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=quest \
    exp_name=final_eval_2 \
    variant_name=quest_block_16_ds_2 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    seed=2 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_ae_final/block_16_ds_2/2/stage_1





sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=quest \
    exp_name=final_eval_2 \
    variant_name=quest_block_16_ds_4 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_ae_final/block_16_ds_4/0/stage_1


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=quest \
    exp_name=final_eval_2 \
    variant_name=quest_block_16_ds_4 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    seed=1 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_ae_final/block_16_ds_4/1/stage_1


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=quest \
    exp_name=final_eval_2 \
    variant_name=quest_block_16_ds_4 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    seed=2 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_ae_final/block_16_ds_4/2/stage_1












# DP


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=diffusion_policy \
    exp_name=final_eval_2 \
    variant_name=diffusion_policy_block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/diffusion_policy/dp_final/block_16/0/stage_1



sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=diffusion_policy \
    exp_name=final_eval_2 \
    variant_name=diffusion_policy_block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    seed=1 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/diffusion_policy/dp_final/block_16/1/stage_1


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=diffusion_policy \
    exp_name=final_eval_2 \
    variant_name=diffusion_policy_block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    seed=2 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/diffusion_policy/dp_final/block_16/2/stage_1




# VQ BeT

sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=final_eval_2 \
    variant_name=bet_block_5_beta_0.5 \
    training.use_tqdm=false \
    algo.skill_block_size=5 \
    algo.beta=0.5 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_final/block_5_beta_0.5/0/stage_1


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=final_eval_2 \
    variant_name=bet_block_5_beta_0.5 \
    training.use_tqdm=false \
    algo.skill_block_size=5 \
    algo.beta=0.5 \
    seed=1 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_final/block_5_beta_0.5/1/stage_1


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=final_eval_2 \
    variant_name=bet_block_5_beta_0.5 \
    training.use_tqdm=false \
    algo.skill_block_size=5 \
    algo.beta=0.5 \
    seed=2 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_final/block_5_beta_0.5/2/stage_1









sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=final_eval_2 \
    variant_name=bet_block_5_beta_0.1 \
    training.use_tqdm=false \
    algo.skill_block_size=5 \
    algo.beta=0.1 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_final/block_5_beta_0.1/0/stage_1


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=final_eval_2 \
    variant_name=bet_block_5_beta_0.1 \
    training.use_tqdm=false \
    algo.skill_block_size=5 \
    algo.beta=0.1 \
    seed=1 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_final/block_5_beta_0.1/1/stage_1


sbatch slurm/run_rtx6000.sbatch python evaluate.py --config-name=evaluate.yaml \
    task=metaworld_ml45_prise \
    algo=bet \
    exp_name=final_eval_2 \
    variant_name=bet_block_5_beta_0.1 \
    training.use_tqdm=false \
    algo.skill_block_size=5 \
    algo.beta=0.1 \
    seed=2 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/bet_final/block_5_beta_0.1/2/stage_1






