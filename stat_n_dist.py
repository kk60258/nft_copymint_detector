import pandas as pd
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# file = 'D:\\Downloads\\NFT\\attack_result.csv'
file = 'D:\\github\\detect_copymint\\temp_0810_attack\\attack_result.csv'
out = os.path.join(os.path.dirname(file), 'stat.csv')
histogram_dir = f'{os.path.dirname(file)}_histogram'
pathlib.Path(histogram_dir).mkdir(parents=True, exist_ok=True)
out_lines = []

df = pd.read_csv(file)
# iterate over each group
df_inf = df[df['min_n'] == np.inf]
for group_name, df_group in df_inf.groupby('name'):
    print(f'{group_name} has inf')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=["min_n", "dist"], how="all", inplace=True)

df.head(3)

def histogram(x, bins, name, color='b', label='x'):
    plt.hist(x, bins=bins, color=color, label=label)
    plt.xticks(bins, rotation=60)
    plt.title(name)
    plt.savefig(os.path.join(histogram_dir, f'{name}.jpg'))
    plt.clf()


def cal_confidence_interval(df, z=1.96, name=''):
    df = df.dropna()
    n = len(df)
    mean = df.mean()
    std = df.std()

    interval = z * std / math.sqrt(n)
    color = 'b' if 'dist' in name else 'g'
    # bins_ori = [mean - 2*std, mean - 1*std, mean, mean + std, mean + 2*std]
    # bins_mask = [True if n > b > 0 else False for b in bins_ori]
    # bins = [0] + [b for b in bins_ori if b > 0]
    # bins = [b for b in bins if b < n] + [n]
    bins = [0, 20, 50, 100, 500, 1000, max(n, 1001)]
    # bins = list(range(0, n, n//7))
    print(bins)
    histogram(df.to_numpy(), bins[:-1], name, color=color)

    frequency, bins_edge = np.histogram(df.to_numpy(), bins=bins)
    print(frequency)
    # f = []
    # idx = 0
    # for b in bins_mask:
    #     if not b:
    #         f.append('-')
    #     else:
    #         f.append(frequency[idx])
    #         idx += 1
    # f.extend(list(frequency[idx:]))

    return (mean - interval, mean + interval, mean, std, n, frequency, bins)

min_n = confidence_interval = cal_confidence_interval(df['min_n'], name='all_n')
dist = confidence_interval = cal_confidence_interval(df['dist'], name='all_dist')
s = f'{"All":<30}, [{min_n[0]:.2f}-{min_n[1]:.2f}], {min_n[2]:.2f}, {min_n[3]:.2f}, {min_n[4]}, {min_n[5][0]}, {min_n[5][1]}, {min_n[5][2]}, {min_n[5][3]}, {min_n[5][4]}, {min_n[5][5]},' \
    f'[{dist[0]:.2f}-{dist[1]:.2f}], {dist[2]:.2f}, {dist[3]:.2f}, {dist[4]}, {dist[5][0]}, {dist[5][1]}, {dist[5][2]}'
out_lines.append(s)
print(s)

# iterate over each group
for group_name, df_group in df.groupby('attack'):
    min_n = confidence_interval = cal_confidence_interval(df_group['min_n'], name=f'{group_name}_n')
    dist = confidence_interval = cal_confidence_interval(df_group['dist'], name=f'{group_name}_dist')
    s = f'{group_name:<30}, [{min_n[0]:.2f}-{min_n[1]:.2f}], {min_n[2]:.2f}, {min_n[3]:.2f}, {min_n[4]}, {min_n[5][0]}, {min_n[5][1]}, {min_n[5][2]}, {min_n[5][3]}, {min_n[5][4]}, {min_n[5][5]},' \
        f'[{dist[0]:.2f}-{dist[1]:.2f}], {dist[2]:.2f}, {dist[3]:.2f}, {dist[4]}, {dist[5][0]}, {dist[5][1]}, {dist[5][2]}'
    out_lines.append(s)
    print(s)
# bins = [0, 20, 50, 100, 500, 1000, n]
n_bins = min_n[6]
dist_bins = dist[6]
title = f'attack_type, n_interval, n_mean, n_std, n_size, n_{n_bins[0]}~{n_bins[1]}, n_{n_bins[1]}~{n_bins[2]}, n_{n_bins[2]}~{n_bins[3]},' \
        f' n_{n_bins[3]}~{n_bins[4]}, n_{n_bins[4]}~{n_bins[5]}, n_{n_bins[5]}~{n_bins[6]}, dist_interval, dist_mean, dist_std, dist_size,' \
        f' dist_{dist_bins[0]}~{dist_bins[1]}, dist_{dist_bins[1]}~{dist_bins[2]}, dist_{dist_bins[2]}~{dist_bins[3]}\n'
print(title)
with open(out, 'w') as f:
    f.writelines(title)
    f.writelines('\n'.join(out_lines))