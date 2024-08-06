import os
import json
import numpy as np

directories = [
    "/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/evaluate/libero/LIBERO_90/act_policy/act_d256/block_16",
    "/storage/home/hcoda1/0/amete7/p-agarg35-0/PRIMO/experiments/evaluate/libero/LIBERO_90/bc_transformer_policy/bctrans_d256/block_10"
]

include = 'stage_1'
scores = {}

for directory in directories:
    success_rates = []
    for root, dirs, files in os.walk(directory):
        if 'data.json' in files:
            if include not in root:
                continue
            with open(os.path.join(root, 'data.json'), 'r') as f:
                data_dict = json.load(f)
            rollout_success_rate = data_dict['rollout_success_rate']
            success_rates.append(list(rollout_success_rate.values()))
    name = directory.split('/')[-3]
    success_rates = np.array(success_rates)
    print(f'{name}: mean: {success_rates.mean()}, std: {success_rates.mean(axis=1).std()}')
    scores[name] = success_rates

# save scores to file
np.savez('act_bc_multi.npz', **scores)