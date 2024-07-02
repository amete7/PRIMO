# python train.py --config-name=train_autoencoder.yaml \
#     task=metaworld_ml45_prise \
#     algo=bet \
#     training.use_amp=true \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=5 \
#     algo.beta=0.5 \
#     seed=0 \
#     logging.mode=disabled


skill_block_size=5
betas=(0.1 0.5)
seeds=(0 1 2)

for seed in ${seeds[@]}; do
    for beta in ${betas[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_ml45_prise_fewshot \
            algo=bet \
            exp_name=bet_fs \
            variant_name=block_${skill_block_size}_beta_${beta} \
            training.use_tqdm=false \
            training.use_amp=true \
            training.save_all_checkpoints=true \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.skill_block_size=${skill_block_size} \
            algo.beta=${beta} \
            training.auto_continue=true \
            seed=${seed}
    done
done

