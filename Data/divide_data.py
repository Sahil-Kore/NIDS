import pandas as pd
import os 
import sys

import glob

files = glob.glob("../Data/dhoogla/cicids2017/versions/3/*.parquet")
df_list = []
for file in files:
    df = pd.read_parquet(file)
    df_list.append(df)

final_df = pd.concat(df_list , ignore_index= True )
final_df["Label"].value_counts()

#keeping the bot class examples aside for the test set 
#the rest of the data will be transformed for a binary classification 

test_df = final_df[final_df["Label"] == "Bot"].copy()
test_df["Label"] = 1

train_val_df = final_df[final_df["Label"] != "Bot"].copy()

train_val_df["Label"] = (train_val_df["Label"] != "Benign").astype(int)

print(test_df["Label"].value_counts())
print(train_val_df["Label"].value_counts())
 

test_df.to_parquet("../Data/test.parquet" , index = False)
train_val_df.to_parquet("../Data/train_val.parquet" , index = False)