import pandas as pd
import numpy as np
import glob

import seaborn as sns
import matplotlib.pyplot as plt

all_files = glob.glob("../../Data/dhoogla/cicids2017/versions/3/*.parquet")
df_list = []

for file in all_files:
    df_sample = pd.read_parquet(
        file, 
    )
    
    df_list.append(df_sample)

df = pd.concat(df_list, axis=0, ignore_index=True)


print(df.describe())


df["Label"].value_counts()
#there are 15 unique classes 

#we can merge 6 classes with the least amount of examples to one class called "malicious" so that in when we use tabpfn or lightgbm it does not causea ny errors
