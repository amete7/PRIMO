task="metaworld_ml45_prise"
algo="quest"
exp_name="quest_cs_1024"
stage=1
variant_names="
block_16_ds_2
block_16_ds_4
"
downsample_factors=(2 4)
# variant_names="
# block_16_ds_2_l1_1 
# block_16_ds_2_l1_10 
# block_16_ds_2_l1_100 
# block_16_ds_4_l1_1 
# block_16_ds_4_l1_10 
# block_16_ds_4_l1_100
# "
seeds=(0 1 2 4 5)


for seed in ${seeds[@]}; do
    for downsample_factor in ${downsample_factors[@]}; do
        variant_name=block_16_ds_${downsample_factor}
        # echo $variant_name
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            task=$task \
            algo=$algo \
            exp_name=${exp_name}_woke \
            variant_name=${variant_name} \
            stage=$stage \
            training.use_tqdm=false \
            seed=$seed \
            algo.skill_block_size=16 \
            algo.downsample_factor=${downsample_factor} \
            checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/${algo}/${exp_name}/${variant_name}/${seed}/stage_${stage}/
    done
done
