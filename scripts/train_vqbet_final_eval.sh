# sbatch slurm/run_rtx6000.sbatch python evaluate.py \
#     task=metaworld_ml45_prise_fewshot \
#     algo=bet \
#     training.use_amp=true \
#     train_dataloader.num_workers=6 \
#     algo.skill_block_size=5 \
#     seed=0 \
#     logging.mode=disabled \
#     task.env_runner.debug=true



sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    stage=1 \
    seed=0


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    stage=1 \
    seed=1


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    stage=1 \
    seed=2









sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    stage=1 \
    seed=0


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    stage=1 \
    seed=1


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    stage=1 \
    seed=2





























sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    stage=2 \
    seed=0


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    stage=2 \
    seed=1


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.5 \
    training.use_tqdm=false \
    stage=2 \
    seed=2









sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    stage=2 \
    seed=0


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    stage=2 \
    seed=1


sbatch slurm/run_rtx6000.sbatch python evaluate.py \
    task=metaworld_ml45_prise_fewshot \
    algo=bet \
    exp_name=bet_final_2 \
    variant_name=block_5_beta_0.1 \
    training.use_tqdm=false \
    stage=2 \
    seed=2






