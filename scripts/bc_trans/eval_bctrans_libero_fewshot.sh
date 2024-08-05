seeds=(0 1 2 3 4 6 8 12 21)

for seed in ${seeds[@]}; do
    sbatch slurm/run_rtx6000.sbatch python evaluate.py \
        task=libero_long_fewshot \
        algo=bc_transformer \
        exp_name=bctrans_d256 \
        variant_name=block_10 \
        stage=2 \
        training.use_tqdm=false \
        algo.embed_dim=256 \
        seed=$seed
done

# python evaluate.py \
#     task=libero_long_fewshot \
#     algo=bc_transformer \
#     exp_name=bctrans_d256_latency \
#     variant_name=block_10 \
#     stage=2 \
#     training.use_tqdm=false \
#     make_unique_experiment_dir=true \
#     algo.embed_dim=256 \
#     checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/libero/LIBERO_90/bc_transformer_policy/bctrans_d256/block_10/0/stage_1/multitask_model_epoch_0025.pth \
#     seed=0

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