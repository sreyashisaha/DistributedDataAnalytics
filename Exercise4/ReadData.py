import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

rows = []
for file in tqdm(os.listdir('dataset')):
    for line in open(f"dataset/{file}", 'r'):
        features = line.split()
        target = features[0]
        row = {feature.split(':')[0]: feature.split(':')[1] for feature in features[1:]}
        row['target'] = target
        rows.append(row)

df = pd.DataFrame(rows)
new_columns = {int(col) for col in df.columns if col != 'target'}
new_columns = list(map(str, new_columns)) + ['target']
df = df.reindex(new_columns, axis=1)
df = df.replace(np.nan, 0)
df.to_csv('/Users/sreyashisaha/Desktop/Semester1/DDA/practical/Week5/virus.csv',index=False)

