
sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_noamp/block_16/0/stage_1 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d512_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d512_noamp/block_16/0/stage_1 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_noamp/block_32/0/stage_1 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_kl100_noamp/block_16/0/stage_1 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d512_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d512_kl100_noamp/block_16/0/stage_1 \
    seed=0

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_kl100_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_kl100_noamp/block_32/0/stage_1 \
    seed=0


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_noamp/block_16/0/stage_1 \
    seed=1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d512_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d512_noamp/block_16/0/stage_1 \
    seed=1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_noamp/block_32/0/stage_1 \
    seed=1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_kl100_noamp/block_16/0/stage_1 \
    seed=1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d512_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d512_kl100_noamp/block_16/0/stage_1 \
    seed=1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_kl100_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_kl100_noamp/block_32/0/stage_1 \
    seed=1


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_noamp/block_16/0/stage_1 \
    seed=2

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d512_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d512_noamp/block_16/0/stage_1 \
    seed=2

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_noamp/block_32/0/stage_1 \
    seed=2

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_kl100_noamp/block_16/0/stage_1 \
    seed=2

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d512_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d512_kl100_noamp/block_16/0/stage_1 \
    seed=2

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_kl100_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_kl100_noamp/block_32/0/stage_1 \
    seed=2


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_noamp/block_16/0/stage_1 \
    seed=3

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d512_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d512_noamp/block_16/0/stage_1 \
    seed=3

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_noamp/block_32/0/stage_1 \
    seed=3

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_kl100_noamp/block_16/0/stage_1 \
    seed=3

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d512_kl100_noamp \
    variant_name=block_16 \
    training.use_tqdm=false \
    algo.skill_block_size=16 \
    algo.embed_dim=512 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d512_kl100_noamp/block_16/0/stage_1 \
    seed=3

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise \
    algo=act_policy \
    exp_name=eval_act_d256_kl100_noamp \
    variant_name=block_32 \
    training.use_tqdm=false \
    algo.skill_block_size=32 \
    algo.embed_dim=256 \
    algo.kl_weight=100 \
    checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/act_d256_kl100_noamp/block_32/0/stage_1 \
    seed=3


# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     algo=act_policy \
#     exp_name=act_d512_noamp \
#     variant_name=block_32 \
#     training.use_tqdm=false \
#     training.use_amp=false \
#     training.save_all_checkpoints=true \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=32 \
#     algo.embed_dim=512 \
#     training.n_epochs=200 \
#     training.resume=true \
#     seed=0

# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     algo=act_policy \
#     exp_name=act_d512_kl100_noamp \
#     variant_name=block_32 \
#     training.use_tqdm=false \
#     training.use_amp=false \
#     training.save_all_checkpoints=true \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=32 \
#     algo.embed_dim=512 \
#     algo.kl_weight=100 \
#     training.n_epochs=200 \
#     training.resume=true \
#     seed=0


# for debugging
# python train.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     algo=act_policy \
#     exp_name=act_debug_noamp \
#     variant_name=block_16 \
#     training.use_tqdm=true \
#     training.use_amp=false \
#     training.save_all_checkpoints=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=true \
#     algo.skill_block_size=16 \
#     algo.frame_stack=1 \
#     training.n_epochs=200 \
#     training.resume=true \
#     logging.mode=disabled \
#     training.do_profile=True \
#     task.env_runner.debug=true \
#     seed=0