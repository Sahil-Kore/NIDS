import pandas as pd
import numpy as np
import glob

import seaborn as sns
import matplotlib.pyplot as plt

all_files = glob.glob("../../Data/dhoogla/cicids2017/versions/3/*.parquet")
df_list = []
sample_frac = 0.01

print("Starting to sample files...")
for file in all_files:
    df_sample = pd.read_parquet(
        file, 
    )
    
    df_sample = df_sample.sample(frac= sample_frac , random_state=42)
    
    df_list.append(df_sample)

df = pd.concat(df_list, axis=0, ignore_index=True)

print(df)
print(df.isnull().sum())

columns = df.columns
