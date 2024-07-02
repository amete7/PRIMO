# python train.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     training.use_amp=false \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2

skill_block_size=16
downsample_factors=(2 4)
seeds=(0 1 2)

for downsample_factor in ${downsample_factors[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_ml45_prise_fewshot \
            exp_name=quest_cs_1024_fs \
            variant_name=block_${skill_block_size}_ds_${downsample_factor} \
            training.use_tqdm=false \
            training.save_all_checkpoints=true \
            training.use_amp=false \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.skill_block_size=${skill_block_size} \
            algo.downsample_factor=$downsample_factor \
            training.auto_continue=true \
            seed=$seed
    done
done




