from argparse import ArgumentParser
import os
import json
import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})


def main():
    parser = ArgumentParser()
    parser.add_argument('data_dirs', nargs='*')
    parser.add_argument('--labels', nargs='*')
    # parser.add_argument('--bold_best', action='store_true')
    parser.add_argument('--include')
    parser.add_argument('--exclude')
    args = parser.parse_args()

    # assert len(args.data_dirs) == len(args.labels)


    # mean_success_rate = {}
    # std_errors = {}


    # prefix = os.path.commonprefix(args.data_dirs)

    # labels = [data_dir[len(prefix):] for data_dir in args.data_dirs]
    # medians, stds = [], []
    data = {}
    for data_dir in args.data_dirs:
        # algo_data = {}
        for root, dirs, files in os.walk(data_dir):
            seed_key = root[len(data_dir):]
            if 'data.json' in files:
                if args.include is not None and args.include not in root or args.exclude is not None and args.exclude in root:
                    # print('skipping', os.path.join())
                    continue
                with open(os.path.join(root, 'data.json'), 'r') as f:
                    data_dict = json.load(f)
                rollout_success_rate = data_dict['rollout_success_rate']
                for env_name in rollout_success_rate:
                    env_data = data.get(env_name, {})
                    seed_data = env_data.get(seed_key, [])
                    # variant_data = seed_data.get(data_dir, [])


                    seed_data.append(rollout_success_rate[env_name])
                    # seed_data[data_dir] = variant_data
                    env_data[seed_key] = seed_data
                    data[env_name] = env_data
    # breakpoint()

    env_avgs = {}
    env_std = {}
    avgs = []
    all_final = []
    for env_name in rollout_success_rate:
        env_data = data[env_name]
        per_seed_max = []
        for seed_key in env_data:
            per_seed_max.append(max(env_data[seed_key])*100)
        all_final.append(per_seed_max)
        avg = np.mean(per_seed_max)
        std = np.std(per_seed_max)
        env_avgs[env_name] = avg
        env_std[env_name] = std
        avgs.append(avg)

    print(env_avgs)
    print(env_std)
    print(np.mean(avgs))
    all_final = np.array(all_final)
    print(all_final.mean(axis=0).std())
    
    # breakpoint()
                    # print(root)
                    # print(root[len(data_dir):])

                    # algo_env_data = algo_data.get(env_name, [])
                    # algo_env_data.append(rollout_success_rate[env_name])
                    # algo_data[env_name] = algo_env_data
        
        # data[data_dir] = algo_data
        # algo_medians = {env_name: np.median(algo_data[env_name]) for env_name in algo_data}
        # medians.append(algo_medians)
        # algo_std_errors = {env_name: np.std(algo_data[env_name]) / np.sqrt(len(algo_data[env_name])) for env_name in algo_data}
        # stds.append(algo_std_errors)

    # env_names = list(medians[0].keys())
    # env_names.sort()
    
    

if __name__ == '__main__':
    main()