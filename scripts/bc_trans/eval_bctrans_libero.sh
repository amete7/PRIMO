seeds=(0 1)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python evaluate.py \
        task=libero_90 \
        algo=bc_transformer \
        exp_name=bctrans_d256 \
        variant_name=block_10 \
        stage=1 \
        training.use_tqdm=false \
        algo.embed_dim=256 \
        seed=$seed
done


# python train.py --config-name=train_prior.yaml \
#     task=libero_90 \
#     algo=bc_transformer \
#     exp_name=bctrans_d256 \
#     variant_name=block_10 \
#     training.use_tqdm=false \
#     training.use_amp=false \
#     training.save_all_checkpoints=true \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.embed_dim=256 \
#     training.n_epochs=100 \
#     algo.policy.image_aug_factory=null \
#     rollout.interval=25 \
#     training.save_interval=1 \
#     training.resume=true \
#     train_dataloader.batch_size=64 \
#     seed=0