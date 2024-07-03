vocab_sizes=(100 150 200 250 300)
seeds=(0 1 2)

for vocab_size in ${vocab_sizes[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_ml45_prise \
            algo=prise \
            exp_name=initial \
            variant_name=prior_bpe_vocab_size_${vocab_size} \
            make_unique_experiment_dir=false \
            training.use_tqdm=false \
            training.save_all_checkpoints=true \
            training.use_amp=false \
            training.grad_clip=10 \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            algo.decoder_loss_coef=0.01 \
            algo.decoder_type=gmm \
            algo.vocab_size=${vocab_size} \
            seed=$seed \
            checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/prise/initial/bpe_vocab_size_${vocab_size}/${seed}/stage_0/multitask_model_bpe.pth
            # training.resume=true 
    done
done
