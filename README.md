# PRIMO
Offical code for PRIMO: Towards Robot Learning with Primitive Motion

1. `squeue -u <username>` to check the status of the job.
2. `scancel <job_id>` to cancel the job.
3. `pace-check-queue -c gpu-a100` to check the queue for A100 GPUs.
4. `pace=quota` to check the quota.
5. `sbatch slurm/train.sbatch python libero/lifelong/skill_policy.py` to use inferno-paid A100-50hrs for training. (A100 is faster than RTX6000)
6. `sbatch slurm/eval.sbatch python libero/lifelong/skill_policy_eval.py` to use embers-free RTX6000-8hrs for eval. (RTX6000 is suitable for rendering)
7. `salloc -A gts-agarg35 -N1 --mem-per-gpu=32G -q embers -t8:00:00 --gres=gpu:V100:1` to start a job on embers-free RTX6000-8hrs.
8. `salloc -A gts-agarg35 -N1 --mem-per-gpu=32G -q inferno -t50:00:00 --gpus=V100:1` --constraint V100-32GB to start a job on inferno-paid V100-32GB-50hrs.