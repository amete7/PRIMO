python train.py --config-name=train_autoencoder.yaml \
    data_prefix=/storage/coda1/p-agarg35/0/shared/quest/data/ \
    output_prefix=/storage/home/hcoda1/1/awilcox31/scratch/primo/results/ \
    logging.mode=disabled


python train.py --config-name=train_prior.yaml \
    data_prefix=/storage/coda1/p-agarg35/0/shared/quest/data/ \
    output_prefix=/storage/home/hcoda1/1/awilcox31/scratch/primo/results/ \
    logging.mode=disabled











# Batch runs


sbatch slurm/run_v100_fixed.sbatch python train.py --config-name=train_autoencoder.yaml \
    data_prefix=/storage/coda1/p-agarg35/0/shared/quest/data/ \
    output_prefix=/storage/home/hcoda1/1/awilcox31/scratch/primo/results/ \
    training.use_tqdm=false \
    exp_name=implementing_1 \
    make_unique_experiment_dir=false


    

