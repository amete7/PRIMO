task="metaworld_ml45_prise"
algo="prise"
exp_name="initial"
stage=1
variant_name="prior_bpe_vocab_size_200"
# variant_names="
# block_16_ds_2_l1_1 
# block_16_ds_2_l1_10 
# block_16_ds_2_l1_100 
# block_16_ds_4_l1_1 
# block_16_ds_4_l1_10 
# block_16_ds_4_l1_100
# "
checkpoints=(0010)
seeds=(0)


for seed in ${seeds[@]}; do
    for checkpoint in ${checkpoints[@]}; do
        # sbatch slurm/run_rtx6000.sbatch python evaluate.py \
        #     task=$task \
        #     algo=$algo \
        #     exp_name=${exp_name}_early_sweep \
        #     variant_name=${variant_name}_${checkpoint} \
        #     stage=$stage \
        #     training.use_tqdm=false \
        #     seed=$seed \
        #     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/${algo}/${exp_name}/${variant_name}/${seed}/stage_${stage}/multitask_model_epoch_${checkpoint}.pth

        python evaluate.py \
            task=$task \
            algo=$algo \
            stage=$stage \
            training.use_tqdm=true \
            seed=$seed \
            rollout.rollouts_per_env=1 \
            make_unique_experiment_dir=true \
            checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/${algo}/${exp_name}/${variant_name}/${seed}/stage_${stage}/multitask_model_epoch_${checkpoint}.pth
    done
done
