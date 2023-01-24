import pandas as pd
import numpy as np

df = pd.read_csv('6000_samples_data.csv')

df = df.dropna(axis=0, subset=['abstract'])

df['abstract'] = df['abstract'].replace('-', np.nan)
df = df.dropna(axis=0, subset=['abstract'])

df.to_csv('6000_samples_data_normalised.csv',index=False)
