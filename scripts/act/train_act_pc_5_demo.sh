

blocks=(4 8)
seeds=(0 1)

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_pc_mt50 \
            algo=act_pc \
            exp_name=act_5_demo \
            variant_name=pc_block_${block} \
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
        
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_pc_mt50 \
            algo=act_pc \
            exp_name=act_5_demo \
            variant_name=pc_no_aug_block_${block} \
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
            algo.policy.aug_factory=null \
            seed=${seed}
        
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_pc_mt50 \
            algo=act_pc_no_lowdim \
            exp_name=act_5_demo \
            variant_name=pc_no_lowdim_block_${block} \
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
        
        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_pc_mt50 \
            algo=act_pc_no_lowdim \
            exp_name=act_5_demo \
            variant_name=pc_no_lowdim_no_aug_block_${block} \
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
            algo.policy.aug_factory=null \
            seed=${seed}

        


        sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_prior.yaml \
            task=metaworld_dp3_mt50 \
            algo=act_pc \
            exp_name=act_5_demo \
            variant_name=dp3_block_${block} \
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
            algo.policy.aug_factory=null \
            seed=${seed}
    done
done




