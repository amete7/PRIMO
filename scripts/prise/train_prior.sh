python train.py --config-name=train_prior.yaml \
    task=metaworld_ml45_prise \
    algo=prise \
    train_dataloader.num_workers=6 \
    algo.decoder_loss_coef=1 \
    algo.vocab_size=200 \
    seed=0 \
    checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/prise/initial/bpe_vocab_size_200/0/stage_0/multitask_model_bpe.pth

