from argparse import ArgumentParser
import os
import json
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# Data
# methods = ['ResNet-T', 'ACT', 'Diffusion\nPolicy', 'PRISE', 'VQ-BeT', 'Ours']
# mean_success_rate = [15.4, 46.6, 75.1, 54.4, 81.4, 89.8]
# error = [0, 0.5, 0.5, 0.5, 0.5, 0.5]  # Example error values

# Colors (more aesthetic)
DEFAULT_COLORS = ['#FF6F61', '#6B5B95', '#88B04B', '#F9B8B8', '#92A8D1', '#607D8B']

# Create plot
# fig, ax = plt.subplots(figsize=(8, 6))

# # Plot bars with error bars
# bars = ax.bar(methods, mean_success_rate, color=colors, yerr=error, capsize=5)

# # Add text annotations
# for i,bar in enumerate(bars):
#     height = bar.get_height()
#     ax.annotate(f'{height:.1f}',
#                 xy=(bar.get_x() + bar.get_width() / 2, height + error[i]/2 + 1.5),
#                 xytext=(0, 3),  # 3 points vertical offset
#                 textcoords="offset points",
#                 ha='center', va='bottom',
#                 fontsize=18)  # Increased font size

# # Labeling
# # ax.set_xlabel('Methods', fontsize=16)
# ax.set_ylabel('Mean Success Rate (%)', fontsize=18)
# # ax.set_title('Multitask-IL LIBERO-90', fontsize=16)
# ax.set_ylim(0, 100)

# # Customize the plot
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.grid(True, linestyle='--', alpha=0.7)
# ax.set_axisbelow(True)

# # Increase font size of x-tick labels
# plt.xticks(rotation=0, fontsize=18)
# plt.yticks(fontsize=18)

# # Show the plot
# plt.tight_layout()
# plt.savefig('lib_multi.pdf')
# plt.close()

def main():
    parser = ArgumentParser()
    parser.add_argument('data-dirs', nargs='*')
    parser.add_argument('--labels', nargs='*')
    parser.add_argument('--colors', nargs='*')
    parser.add_argument('--xlabel')
    parser.add_argument('--ylabel')
    parser.add_argument('--title')
    parser.add_argument('--fname')
    parser.add_argument('--font-size', default=16, type=int)
    parser.add_argument('--font-size-xticks', default=18, type=int)
    parser.add_argument('--font-size-yticks', default=18, type=int)
    parser.add_argument('--font-size-xlabel', default=18, type=int)
    parser.add_argument('--font-size-ylabel', default=18, type=int)
    parser.add_argument('--font-size-annotation', default=18, type=int)
    parser.add_argument('--font-size-title', default=18, type=int)
    parser.add_argument('--ylim-low', default=0)
    parser.add_argument('--ylim-high', default=100)
    args = parser.parse_args()

    assert args.labels is None or len(args.labels) == len(args.data_dirs), \
        'Must give no labels or same number of labels and data dirs'
    labels = args.labels if args.labels is not None else args.data_dirs

    assert args.colors is None or len(args.colors) >= len(args.data_dirs), \
        'Must give no colors or at least as many colors as data dirs'
    colors = args.colors if args.colors is not None else DEFAULT_COLORS

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
    matplotlib.rc('font', **font)

    mean_success_rate = []
    std_error = []
    for data_dir in args.data_dirs:
        data = []
        for root, dirs, files in os.walk(data_dir):
            if 'data.json' in files:
                with open(os.path.join(root, 'data.json'), 'r') as f:
                    data_dict = json.load(f)
                data.append[data_dict['rollout']['overall_success_rate']]
        mean_success_rate.append(np.mean(data))
        std_error.append(np.std(data) / np.sqrt(len(data)))

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bars with error bars
    bars = ax.bar(labels, mean_success_rate, color=colors, yerr=std_error, capsize=5)

    # Add text annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std_error[i]/2 + 1.5),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=args.font_size_annotation)  # Increased font size

    # Labeling
    ax.set_xlabel(args.xlabel, fontsize=args.font_size_xlabel)
    ax.set_ylabel(args.ylabel, fontsize=args.font_size_ylabel)
    ax.set_title(args.title, fontsize=args.font_size_title)
    ax.set_ylim(args.ylim_low, args.ylim_high)

    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Increase font size of x-tick labels
    plt.xticks(rotation=0, fontsize=args.font_size_xticks)
    plt.yticks(fontsize=args.font_size_yticks)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'plots/{args.fname}.pdf')
    plt.close()


if __name__ == '__main__':
    main()