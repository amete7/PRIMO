#!/bin/bash

# List of tasks
tasks=(
    "dial-turn"
    "lever-pull"
    "push-back"
    "soccer"
    "stick-pull"
    "bin-picking"
)

# Loop through each task and run the Python script
for task in "${tasks[@]}"; do
    python /tdmpc2/tdmpc2/collect_data.py "task=mw-${task}" "checkpoint=/tdmpc2/checkpoints/mw-${task}-3.pt" "eval_episodes=50" "+data_dir=/tdmpc2/datasets"
done
