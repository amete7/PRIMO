
blocks=(4 8)
seeds=(0 1)

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_mt50 \
            algo=act \
            exp_name=act_5_demo \
            variant_name=image_block_${block} \
            training.use_tqdm=false \
            training.use_amp=false \
            training.save_all_checkpoints=true \
            train_dataloader.persistent_workers=true \
            train_dataloader.num_workers=6 \
            make_unique_experiment_dir=false \
            algo.skill_block_size=${block} \
            algo.embed_dim=256 \
            training.n_epochs=2000 \
            training.resume=false \
            task.demos_per_env=5 \
            seed=${seed}
    done
done



# python train.py --config-name=train_prior.yaml \
#     task=metaworld_dp3_mt50 \
#     algo=act_pc \
#     training.use_tqdm=true \
#     training.use_amp=false \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=8 \
#     algo.embed_dim=256 \
#     training.n_epochs=2000 \
#     training.resume=true \
#     algo.policy.aug_factory=null \
#     seed=1 \
#     checkpoint_path=./
# /metaworld_pc/MT50/act/act_baseline/dp3_block_8/1/stage_1/multitask_model_final.pth


# rm experiments/metaworld_pc/MT50/act/act_baseline/dp3_block_4/1/stage_1/multitask_model_final.pth
# rm experiments/metaworld_pc/MT50/act/act_baseline/dp3_block_8/1/stage_1/multitask_model_final.pth
# rm experiments/metaworld_pc/MT50/act/act_baseline/pc_block_4/1/stage_1/multitask_model_final.pth
# rm experiments/metaworld_pc/MT50/act/act_baseline/pc_block_8/1/stage_1/multitask_model_final.pth
# rm experiments/metaworld_pc/MT50/act/act_baseline/pc_no_aug_block_4/1/stage_1/multitask_model_final.pth
# rm experiments/metaworld_pc/MT50/act/act_baseline/pc_no_aug_block_8/1/stage_1/multitask_model_final.pth
# rm experiments/metaworld_pc/MT50/act/act_baseline/pc_no_lowdim_block_4/1/stage_1/multitask_model_final.pth
# rm experiments/metaworld_pc/MT50/act/act_baseline/pc_no_lowdim_block_8/1/stage_1/multitask_model_final.pth
# rm experiments/metaworld_pc/MT50/act/act_baseline/pc_no_lowdim_no_aug_block_4/1/stage_1/multitask_model_final.pth
# rm experiments/metaworld_pc/MT50/act/act_baseline/pc_no_lowdim_no_aug_block_8/1/stage_1/multitask_model_final.pth