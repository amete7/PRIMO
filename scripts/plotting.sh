# python scripts/make_bar_plot.py \
#     experiments/metaworld/ML45_PRISE/quest/final_eval_2/quest_block_16_ds_2/



python scripts/generate_per_env_table.py \
    experiments/evaluate/metaworld/ML45_PRISE/diffusion_policy/dp_final_early_sweep/block_16_multitask_model_epoch_0150.pth \
    experiments/evaluate/metaworld/ML45_PRISE/vqbet/bet_final_2/block_5_beta_0.1 \
    experiments/evaluate/metaworld/ML45_PRISE/quest/quest_cs_1024_early_sweep/block_16_ds_2_60 \
    --labels "Diffusion Policy" "VQBeT" "QueST" \
    --filter stage_2


python scripts/generate_per_env_table.py \
    experiments/evaluate/metaworld/ML45_PRISE/diffusion_policy/dp_final_early_sweep/block_16_multitask_model_final.pth \
    /storage/coda1/p-agarg35/0/shared/act_data/eval_act_d512_noamp \
    experiments/evaluate/metaworld/ML45_PRISE/vqbet/bet_woke/block_5_beta_0.5_ \
    experiments/evaluate/metaworld/ML45_PRISE/quest/quest_cs_1024_take_2/block_16_ds_4 \
    --labels "Diffusion Policy" "ACT" "VQBeT" "QueST" \
    --exclude stage_2



python scripts/hacky_prise_compare.py \
    experiments/evaluate/metaworld/ML45_PRISE/diffusion_policy/dp_final_early_sweep/block_16_multitask_model_final.pth \
    /storage/coda1/p-agarg35/0/shared/act_data/eval_act_d512_noamp \
    experiments/evaluate/metaworld/ML45_PRISE/vqbet/bet_woke/block_5_beta_0.5_ \
    experiments/evaluate/metaworld/ML45_PRISE/quest/quest_cs_1024_take_2/block_16_ds_4 \
    --labels "Diffusion Policy" "ACT" "VQBeT" "QueST" \
    --exclude stage_2

