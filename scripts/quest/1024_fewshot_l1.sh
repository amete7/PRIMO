# python train.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     training.use_amp=false \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2


l1s=(10 100)
seeds=(0 1 2 4 5)

for l1 in ${l1s[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
            task=metaworld_ml45_prise_fewshot \
            exp_name=quest_ae_final \
            variant_name=block_16_ds_2_l1_${l1} \
            training.use_tqdm=false \
            training.save_all_checkpoints=true \
            training.use_amp=false \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.skill_block_size=16 \
            algo.downsample_factor=2 \
            algo.l1_loss_scale=$l1 \
            seed=$seed \
            checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/quest/quest_ae_final/block_16_ds_2/${seed}/stage_1/
    done
done








