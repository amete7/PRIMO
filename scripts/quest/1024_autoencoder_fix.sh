# python train.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     training.use_amp=false \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2

# skill_block_size=16
# downsample_factors=(2 4)
# seeds=(0 1 2)
# checkpoints=(10 20 30 40 50 60 70 80 90)

# for downsample_factor in ${downsample_factors[@]}; do
#     for seed in ${seeds[@]}; do
#         sbatch slurm/run_rtx6000.sbatch python fix.py --config-name=train_autoencoder.yaml \
#             task=metaworld_ml45_prise \
#             exp_name=quest_cs_1024 \
#             variant_name=block_${skill_block_size}_ds_${downsample_factor} \
#             training.use_tqdm=false \
#             training.save_all_checkpoints=true \
#             training.use_amp=false \
#             train_dataloader.persistent_workers=true \
#             train_dataloader.num_workers=6 \
#             make_unique_experiment_dir=false \
#             algo.skill_block_size=${skill_block_size} \
#             algo.downsample_factor=$downsample_factor \
#             training.resume=true \
#             seed=$seed
#     done
# done

task="metaworld_ml45_prise_fewshot"
algo="quest"
exp_name="quest_cs_1024"
stage=2
variant_name="block_16_ds_2"
# variant_names="
# block_16_ds_2_l1_1 
# block_16_ds_2_l1_10 
# block_16_ds_2_l1_100 
# block_16_ds_4_l1_1 
# block_16_ds_4_l1_10 
# block_16_ds_4_l1_100
# "
checkpoints=(10 20 30 40 50 60 70 80 90)
seeds=(0 1 2)


for seed in ${seeds[@]}; do
    for checkpoint in ${checkpoints[@]}; do
        sbatch slurm/run_rtx6000.sbatch python fix.py --config-name=train_autoencoder.yaml \
            task=${task} \
            exp_name=${exp_name} \
            variant_name=${variant_name} \
            training.use_tqdm=false \
            training.save_all_checkpoints=true \
            training.use_amp=false \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.skill_block_size=16 \
            algo.downsample_factor=2 \
            training.resume=true \
            seed=$seed \
            checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/${algo}/${exp_name}/${variant_name}/${seed}/stage_2/multitask_model_epoch_00${checkpoint}.pth
    done
done



