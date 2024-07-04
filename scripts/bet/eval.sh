task="metaworld_ml45_prise"
algo="bet"
exp_name="bet_final_2"
stage=1
# checkpoints=(10 20 30 40 50 60 70 80 90)
seeds=(0 1 2 3 4)
betas=(0.1 0.5)


for seed in ${seeds[@]}; do
    for beta in ${betas[@]}; do
        variant_name=block_5_beta_${beta}
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            task=$task \
            algo=$algo \
            exp_name=bet_woke \
            variant_name=${variant_name}_${checkpoint} \
            stage=$stage \
            training.use_tqdm=false \
            seed=$seed \
            algo.skill_block_size=5 \
            algo.beta=${beta} \
            checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/${exp_name}/${variant_name}/${seed}/stage_${stage}
    done
done
