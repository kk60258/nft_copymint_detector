import pandas as pd
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# file = 'D:\\Downloads\\NFT\\attack_result.csv'
file = 'D:\\github\\detect_copymint\\temp_0730_attack\\attack_result.csv'
out = os.path.join(os.path.dirname(file), 'large_nn.csv')
out_lines = []

df = pd.read_csv(file)
# iterate over each group
df_inf = df[df['min_n'] != float('inf')]
df_large_n = df_inf[df_inf['min_n'] > 1000]
for group_name, df_group in df_large_n.groupby('name'):
    print(f'{group_name} has large n')
    out_lines.append(group_name)

with open(out, 'w') as f:
    f.writelines('\n'.join(out_lines))