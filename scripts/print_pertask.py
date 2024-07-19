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
    parser.add_argument('--filter')
    args = parser.parse_args()

    prefix = os.path.commonprefix(args.data_dirs)
    # labels = [data_dir[len(prefix):] for data_dir in args.data_dirs]
    for data_dir in args.data_dirs:
        envs_data = {}
        envs_mean = {}
        envs_std = {}
        full_data = []
        for root, dirs, files in os.walk(data_dir):
            if 'data.json' in files:
                if args.filter is not None and args.filter not in root:
                    # print('skipping', os.path.join())
                    continue
                with open(os.path.join(root, 'data.json'), 'r') as f:
                    data_dict = json.load(f)
                rollout_success_rate = data_dict['rollout_success_rate']
                for env_name in rollout_success_rate:
                    env_data = envs_data.get(env_name, [])
                    env_data.append(rollout_success_rate[env_name])
                    envs_data[env_name] = env_data
        for env_name in envs_data:
            envs_mean[env_name] = np.mean(envs_data[env_name])
            envs_std[env_name] = np.std(envs_data[env_name])
            full_data.append(envs_data[env_name])
        print(envs_mean)
        #print(envs_std)
        full_data = np.array(full_data)
        per_seed_mean = np.mean(full_data, axis=0)
        print(per_seed_mean.mean(), per_seed_mean.std())
        # breakpoint()

    # Create plot
    # fig, ax = plt.subplots(figsize=(8, 6))

    # # Plot bars with error bars
    # bars = ax.bar(labels, mean_success_rate, color=colors, yerr=std_error, capsize=5)

    # # Add text annotations
    # for i, bar in enumerate(bars):
    #     height = bar.get_height()
    #     ax.annotate(f'{height:.1f}',
    #                 xy=(bar.get_x() + bar.get_width() / 2, height + std_error[i]/2 + 1.5),
    #                 xytext=(0, 3),  # 3 points vertical offset
    #                 textcoords="offset points",
    #                 ha='center', va='bottom',
    #                 fontsize=args.font_size_annotation)  # Increased font size

    # # Labeling
    # ax.set_xlabel(args.xlabel, fontsize=args.font_size_xlabel)
    # ax.set_ylabel(args.ylabel, fontsize=args.font_size_ylabel)
    # ax.set_title(args.title, fontsize=args.font_size_title)
    # ax.set_ylim(args.ylim_low, args.ylim_high)

    # # Customize the plot
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # ax.set_axisbelow(True)

    # # Increase font size of x-tick labels
    # plt.xticks(rotation=0, fontsize=args.font_size_xticks)
    # plt.yticks(fontsize=args.font_size_yticks)

    # # Show the plot
    # plt.tight_layout()
    # plt.savefig(f'plots/{args.fname}.pdf')
    # plt.close()


if __name__ == '__main__':
    main()