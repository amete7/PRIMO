seeds=(0 1 2)
decoder_loss_coefs=(1 3 10)

for decoder_loss_coef in ${decoder_loss_coefs[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_autoencoder.yaml \
            task=metaworld_ml45_prise \
            algo=prise \
            exp_name=initial \
            variant_name=decoder_loss_${decoder_loss_coef}_no_gc \
            training.use_tqdm=false \
            training.save_all_checkpoints=true \
            training.use_amp=true \
            training.grad_clip=null \
            training.n_epochs=50 \
            training.save_interval=1 \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.decoder_loss_coef=${decoder_loss_coef} \
            training.resume=true \
            seed=$seed
    done
done



