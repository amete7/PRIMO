python scripts/make_bar_plot.py \
    experiments/metaworld/ML45_PRISE/quest/final_eval_2/quest_block_16_ds_2/



python scripts/generate_per_env_table.py \
    experiments/evaluate/metaworld/ML45_PRISE/diffusion_policy/dp_final_early_sweep/block_16_multitask_model_final.pth \
    experiments/evaluate/metaworld/ML45_PRISE/vqbet/bet_final_2/block_5_beta_0.1 \
    experiments/evaluate/metaworld/ML45_PRISE/quest/quest_cs_1024_early_sweep/block_16_ds_2_60 \
    --labels "Diffusion Policy" "VQBeT" "QueST" \
    --filter stage_2

