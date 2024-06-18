
sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=default \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \

sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=grad_norm \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \
    algo.policy.grad_norm_clip=100 \

sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=lr_sch \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \
    algo.policy.use_lr_scheduler=false \

sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=kl_rew_0 \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \
    algo.policy.spt_kldiv_scale=0.0 \

sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=lr \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \
    algo.lr=0.00001 \


# task override


sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=over_default \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \
    override_task=sweep-into-v2 \

sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=over_grad_norm \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \
    algo.policy.grad_norm_clip=100 \
    override_task=sweep-into-v2 \

sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=over_lr_sch \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \
    algo.policy.use_lr_scheduler=false \
    override_task=sweep-into-v2 \

sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=over_kl_rew_0 \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \
    algo.policy.spt_kldiv_scale=0.0 \
    override_task=sweep-into-v2 \

sbatch slurm/run_rtx6000.sbatch python train_ppo.py --config-name=train_ppo.yaml \
    exp_name=over_lr \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    seed=0 \
    algo.lr=0.00001 \
    override_task=sweep-into-v2 \


# debug

python ppo_debug.py --config-name=train_ppo.yaml \
    exp_name=over_kl_rew_0 \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=false \
    seed=0 \
    algo.policy.spt_kldiv_scale=0.0 \
    override_task=sweep-into-v2 \
    checkpoint_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45/ppo/over_kl_rew_0/hand-insert-v2/0/run_000/multitask_model_epoch_1700.pth' \
    debug=true \
    save_video=true \
    logging.mode=disabled

python ppo_debug.py --config-name=train_ppo.yaml \
    exp_name=default \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=false \
    seed=0 \
    checkpoint_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45/ppo/default/hand-insert-v2/0/run_000/multitask_model_epoch_1700.pth' \
    debug=true \
    save_video=true \
    logging.mode=disabled

python ppo_debug.py --config-name=train_ppo.yaml \
    exp_name=kl_rew_0 \
    env.env_name=hand-insert-v2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=false \
    seed=0 \
    algo.policy.spt_kldiv_scale=0.0 \
    checkpoint_path='/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45/ppo/kl_rew_0/hand-insert-v2/0/run_000/multitask_model_epoch_1700.pth' \
    debug=true \
    save_video=true \
    logging.mode=disabled