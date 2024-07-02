seeds=(0 1 2)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
        task=metaworld_ml45_prise \
        algo=prise \
        exp_name=prise_gmm \
        variant_name=decoder_loss_1 \
        training.use_tqdm=false \
        training.save_all_checkpoints=true \
        training.use_amp=false \
        training.grad_clip=10 \
        training.n_epochs=30 \
        training.save_interval=1 \
        train_dataloader.persistent_workers=true \
        train_dataloader.num_workers=6 \
        make_unique_experiment_dir=false \
        algo.decoder_loss_coef=0.01 \
        algo.decoder_type=gmm \
        seed=$seed
done




