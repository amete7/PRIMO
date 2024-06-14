# python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     training.use_amp=false \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=32 \
#     algo.downsample_factor=2 \
#     logging.mode=disabled \
#     seed=0 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_2/0/stage_1/




sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_32_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=2 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_2/0/stage_1/


sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_32_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=4 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_4/0/stage_1/

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_32_ds_8 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=32 \
    algo.downsample_factor=8 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_32_ds_8/0/stage_1/














sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_16_ds_2 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=2 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_2/0/stage_1/

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_16_ds_4 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=4 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_4/0/stage_1/

sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
    task=metaworld_ml45_prise_fewshot \
    exp_name=tune_2_fewshot \
    variant_name=block_16_ds_8 \
    training.use_tqdm=false \
    training.save_all_checkpoints=true \
    training.use_amp=false \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.skill_block_size=16 \
    algo.downsample_factor=8 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/tune_2_prior/block_16_ds_8/0/stage_1/







