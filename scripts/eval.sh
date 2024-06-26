task="metaworld_ml45_prise_fewshot"
algo="quest"
exp_name="quest_final_2.1"
stage=2
variant_names="
    block_16_ds_4_l1_1_offset
    block_16_ds_2_l1_1_offset
    block_16_ds_4_l1_10_offset
    block_16_ds_2_l1_10_offset
    block_16_ds_4_l1_100_offset
    block_16_ds_2_l1_100_offset
    block_16_ds_2_l1_0
    block_16_ds_2_l1_10_decoder_ft
    block_16_ds_2_l1_10_offset_no_decoder_ft
    block_16_ds_2_l1_100_decoder_ft
    block_16_ds_2_l1_100_offset_no_decoder_ft
    block_16_ds_4_l1_0
    block_16_ds_4_l1_10_decoder_ft
    block_16_ds_4_l1_10_offset_no_decoder_ft
    block_16_ds_4_l1_100_decoder_ft
    block_16_ds_4_l1_100_offset_no_decoder_ft
"
seeds=(0 1 2)

for variant_name in $variant_names; do
    for seed in $seeds; do
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            task=$task \
            algo=$algo \
            exp_name=$exp_name \
            variant_name=$variant_name \
            stage=$stage \
            training.use_tqdm=false \
            seed=$seed
    done
done

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_1_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_1_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_1_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_1_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_1_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_1_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2





# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_10_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_10_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_10_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_10_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_10_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_10_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2






# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1

# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_offset \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_0 \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_0 \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_0 \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_10_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

    
# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_10_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_10_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_10_offset_no_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_10_offset_no_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_10_offset_no_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

    
# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_offset_no_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

    
# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_offset_no_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_2_l1_100_offset_no_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_0 \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

    
# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_0 \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_0 \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_10_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

    
# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_10_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_10_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_10_offset_no_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

    
# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_10_offset_no_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_10_offset_no_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

    
# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_decoder_ft \
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 








# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_offset_no_decoder_ft\
#     stage=2 \
#     training.use_tqdm=false \
#     seed=0

    
# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_offset_no_decoder_ft\
#     stage=2 \
#     training.use_tqdm=false \
#     seed=1


# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=quest \
#     exp_name=quest_final_2.1 \
#     variant_name=block_16_ds_4_l1_100_offset_no_decoder_ft\
#     stage=2 \
#     training.use_tqdm=false \
#     seed=2 

    