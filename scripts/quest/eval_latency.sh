task="metaworld_ml45_prise_fewshot"
algo="quest"
exp_name="quest_ae_final"
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
checkpoints=(90)
seeds=(0)


for seed in ${seeds[@]}; do
    for checkpoint in ${checkpoints[@]}; do
        python evaluate.py \
            task=$task \
            algo=$algo \
            exp_name=${exp_name}_latency \
            variant_name=${variant_name}_${checkpoint} \
            stage=$stage \
            training.use_tqdm=false \
            seed=$seed \
            checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/${algo}/${exp_name}/${variant_name}/${seed}/stage_2/multitask_model_epoch_00${checkpoint}.pth
    done
done
