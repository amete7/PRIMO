# python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     training.use_amp=false \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=2 \
#     logging.mode=disabled \
#     seed=0 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_2/0/stage_1/ \
#     algo.l1_loss_scale=10 \
#     task.env_runner.debug=true \
#     rollout.rollouts_per_env=1 \
#     +algo.do_fewshot_embedding_hack=true \
#     +task.dataset.do_fewshot_embedding_hack=true




sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_32_ds_2_emb_hack \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=2 \
    +algo.do_fewshot_embedding_hack=true \
    +task.dataset.do_fewshot_embedding_hack=true \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_2/0/stage_1/


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_32_ds_4_emb_hack \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    +algo.do_fewshot_embedding_hack=true \
    +task.dataset.do_fewshot_embedding_hack=true \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_4/0/stage_1/

# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     exp_name=tune_2_fewshot \
#     variant_name=block_32_ds_8_emb_hack \
#     training.use_tqdm=false \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=8 \
    # +algo.do_fewshot_embedding_hack=true \
    # +task.dataset.do_fewshot_embedding_hack=true \
#     seed=0 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_8/0/stage_1/














sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_16_ds_2_emb_hack \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    +algo.do_fewshot_embedding_hack=true \
    +task.dataset.do_fewshot_embedding_hack=true \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_2/0/stage_1/

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_16_ds_4_emb_hack \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    +algo.do_fewshot_embedding_hack=true \
    +task.dataset.do_fewshot_embedding_hack=true \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_4/0/stage_1/

# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     exp_name=tune_2_fewshot \
#     variant_name=block_16_ds_8_emb_hack \
#     training.use_tqdm=false \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=8 \
    # +algo.do_fewshot_embedding_hack=true \
    # +task.dataset.do_fewshot_embedding_hack=true \
#     seed=0 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_8/0/stage_1/







