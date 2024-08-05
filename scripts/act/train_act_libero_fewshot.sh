
# seeds=(3)

# for seed in ${seeds[@]}; do
#     sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
#         task=libero_long_fewshot \
#         algo=act_policy \
#         exp_name=act_d256 \
#         variant_name=block_16 \
#         training.use_tqdm=false \
#         training.use_amp=false \
#         training.save_all_checkpoints=true \
#         train_dataloader.persistent_workers=true \
#         train_dataloader.num_workers=6 \
#         make_unique_experiment_dir=false \
#         algo.skill_block_size=16 \
#         algo.embed_dim=256 \
#         training.n_epochs=100 \
#         algo.policy.image_aug_factory=null \
#         training.auto_continue=true \
#         seed=$seed
# done

pre_seeds=(3)
fs_seeds=(1 2)

for seed in ${pre_seeds[@]}; do
    for fs_seed in ${fs_seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
            task=libero_long_fewshot \
            algo=act_policy \
            exp_name=act_d256 \
            variant_name=block_16 \
            training.use_tqdm=false \
            training.use_amp=false \
            training.save_all_checkpoints=true \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.skill_block_size=16 \
            algo.embed_dim=256 \
            training.n_epochs=100 \
            algo.policy.image_aug_factory=null \
            checkpoint_path=/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/libero/LIBERO_90/act_policy/act_d256/block_16/${seed}/stage_1/ \
            seed=$((fs_seed*(seed+4)))
    done
done

