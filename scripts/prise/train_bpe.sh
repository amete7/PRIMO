vocab_sizes=(100 150 250 300)
seeds=(0 1 2)

for vocab_size in ${vocab_sizes[@]}; do
    for seed in ${seeds[@]}; do
        sbatch slurm/run_rtx6000.sbatch python train_prise_bpe.py \
            task=metaworld_ml45_prise \
            algo=prise \
            exp_name=initial \
            variant_name=bpe_vocab_size_${vocab_size} \
            make_unique_experiment_dir=false \
            algo.decoder_loss_coef=1 \
            algo.vocab_size=${vocab_size} \
            seed=$seed \
            checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/prise/initial/decoder_loss_1/${seed}/stage_0/multitask_model_epoch_0020.pth
    done
done



