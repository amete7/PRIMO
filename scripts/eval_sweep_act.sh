task="metaworld_ml45_prise_fewshot"
algo="act_policy"
exp_name="act_final"
stage=2
variant_name="block_16_dim_512"
checkpoints="
multitask_model_epoch_0020.pth
multitask_model_epoch_0040.pth
multitask_model_epoch_0060.pth
multitask_model_epoch_0080.pth
multitask_model_epoch_0100.pth
multitask_model_epoch_0120.pth
multitask_model_epoch_0140.pth
multitask_model_epoch_0160.pth
multitask_model_epoch_0180.pth
multitask_model_final.pth
"
seeds=(0 1 2 3 4)

for seed in ${seeds[@]}; do
    for checkpoint in ${checkpoints[@]}; do
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            task=$task \
            algo=$algo \
            exp_name=${exp_name}_sweep \
            variant_name=${variant_name}_${checkpoint} \
            stage=$stage \
            training.use_tqdm=false \
            seed=$seed \
            checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/${exp_name}/${variant_name}/${seed}/stage_2/${checkpoint}
    done
done


# task="metaworld_ml45_prise_fewshot"
# algo="act_policy"
# exp_name="act_final"
# stage=2
# variant_name="block_16_dim_256"
# checkpoints="
# multitask_model_epoch_0020.pth
# multitask_model_epoch_0040.pth
# multitask_model_epoch_0060.pth
# multitask_model_epoch_0080.pth
# multitask_model_epoch_0100.pth
# multitask_model_epoch_0120.pth
# multitask_model_epoch_0140.pth
# multitask_model_epoch_0160.pth
# multitask_model_epoch_0180.pth
# multitask_model_final.pth
# "
# seeds=(0 1 2 3 4)

# for seed in ${seeds[@]}; do
#     for checkpoint in ${checkpoints[@]}; do
#         sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#             task=$task \
#             algo=$algo \
#             exp_name=${exp_name}_sweep \
#             variant_name=${variant_name}_${checkpoint} \
#             stage=$stage \
#             training.use_tqdm=false \
#             seed=$seed \
#             checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/metaworld/ML45_PRISE/act_policy/${exp_name}/${variant_name}/${seed}/stage_2/${checkpoint}
#     done
# done

