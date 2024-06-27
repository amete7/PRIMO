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
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            task=$task \
            algo=$algo \
            exp_name=${exp_name}_early_sweep \
            variant_name=${variant_name}_${checkpoint} \
            stage=$stage \
            training.use_tqdm=false \
            seed=$seed \
            checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/${algo}/${exp_name}/${variant_name}/${seed}/stage_2/multitask_model_epoch_00${checkpoint}.pth
    done
done
