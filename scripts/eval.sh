sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=quest \
    exp_name=quest_final_2.1 \
    variant_name=block_16_ds_4_l1_1_offset \
    stage=2 \
    training.use_tqdm=false \
    seed=0

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=quest \
    exp_name=quest_final_2.1 \
    variant_name=block_16_ds_4_l1_1_offset \
    stage=2 \
    training.use_tqdm=false \
    seed=1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=quest \
    exp_name=quest_final_2.1 \
    variant_name=block_16_ds_4_l1_1_offset \
    stage=2 \
    training.use_tqdm=false \
    seed=2













sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=quest \
    exp_name=quest_final_2.1 \
    variant_name=block_16_ds_2_l1_1_offset \
    stage=2 \
    training.use_tqdm=false \
    seed=0

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=quest \
    exp_name=quest_final_2.1 \
    variant_name=block_16_ds_2_l1_1_offset \
    stage=2 \
    training.use_tqdm=false \
    seed=1

sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=quest \
    exp_name=quest_final_2.1 \
    variant_name=block_16_ds_2_l1_1_offset \
    stage=2 \
    training.use_tqdm=false \
    seed=2