# python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     training.use_amp=false \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2 \
#     algo.l1_loss_scale=1 \
#     algo.policy.autoencoder.codebook_size=512 \
#     algo.policy.policy_prior.offset_layers=2 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_final_2/block_16_ds_2/0/stage_1/ \
#     task.env_runner.debug=true \
#     logging.mode=disabled 


# seeds=(0 1 2)
l1s=(1 10 100)
dss=(2 4)

for l1 in $l1s
do
    for ds in ${dss}
    do
        for seed in {0..2}
        do
            sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
                task=metaworld_ml45_prise_fewshot \
                algo=quest \
                exp_name=quest_final_2.1 \
                variant_name=block_16_ds_${ds}_l1_${l1}_decoder_ft \
                training.use_tqdm=false \
                training.save_all_checkpoints=true \
                training.use_amp=false \
                train_dataloader.persistent_workers=true \
                train_dataloader.num_workers=6 \
                make_unique_experiment_dir=false \
                algo.skill_block_size=16 \
                algo.downsample_factor=${ds} \
                algo.l1_loss_scale=${l1} \
                algo.policy.autoencoder.codebook_size=512 \
                checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_final_2/block_16_ds_${ds}/${seed}/stage_1 \
                seed=${seed}
        done
    done
done



# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_decoder_ft \
#     training.use_tqdm=false \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2 \
#     algo.l1_loss_scale=100 \
#     algo.policy.autoencoder.codebook_size=512 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_final_2/block_16_ds_2/0/stage_1 \
#     seed=0


# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_decoder_ft \
#     training.use_tqdm=false \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2 \
#     algo.l1_loss_scale=100 \
#     algo.policy.autoencoder.codebook_size=512 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_final_2/block_16_ds_2/1/stage_1 \
#     seed=1





# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_decoder_ft \
#     training.use_tqdm=false \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=2 \
#     algo.l1_loss_scale=100 \
#     algo.policy.autoencoder.codebook_size=512 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_final_2/block_16_ds_2/2/stage_1 \
#     seed=2







# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_decoder_ft \
#     training.use_tqdm=false \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=4 \
#     algo.l1_loss_scale=100 \
#     algo.policy.autoencoder.codebook_size=512 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_final_2/block_16_ds_4/0/stage_1 \
#     seed=0



# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_decoder_ft \
#     training.use_tqdm=false \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=4 \
#     algo.l1_loss_scale=100 \
#     algo.policy.autoencoder.codebook_size=512 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_final_2/block_16_ds_4/1/stage_1 \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python train.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_decoder_ft \
#     training.use_tqdm=false \
#     training.save_all_checkpoints=true \
#     training.use_amp=false \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.skill_block_size=16 \
#     algo.downsample_factor=4 \
#     algo.l1_loss_scale=100 \
#     algo.policy.autoencoder.codebook_size=512 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/quest/quest_final_2/block_16_ds_4/2/stage_1 \
#     seed=2








