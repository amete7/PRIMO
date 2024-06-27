task="metaworld_ml45_prise_fewshot"
algo="bet"
exp_name="bet_final_2"
stage=2
variant_name="block_5_beta_0.5"
checkpoints=(10 20 30 40 50 60 70 80 90)
seeds=(0 1 2)


for seed in ${seeds[@]}; do
    for checkpoint in ${checkpoints[@]}; do
        sbatch slurm/run_rtx6000.sbatch python fix.py --config-name=train_autoencoder.yaml \
            task=${task} \
            algo=${algo} \
            exp_name=${exp_name} \
            variant_name=${variant_name} \
            training.use_tqdm=false \
            training.save_all_checkpoints=true \
            training.use_amp=false \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.skill_block_size=5 \
            algo.beta=0.5 \
            training.resume=true \
            seed=$seed \
            checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/vqbet/${exp_name}/${variant_name}/${seed}/stage_2/multitask_model_epoch_00${checkpoint}.pth
    done
done



