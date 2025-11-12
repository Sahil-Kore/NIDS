import os
import pandas as pd 
from glob import glob
from sklearn.model_selection import train_test_split

all_files = glob("./raw_data/*.parquet")

df_list = [pd.read_parquet(file) for file in all_files]

df = pd.concat(df_list , axis = 0 , ignore_index= True)

benign_df = df[df["Label"] == "Benign"].copy()
attack_df = df[df["Label"] != "Benign"].copy()

label_counts = attack_df["Label"].value_counts()
labels_to_replace = label_counts[label_counts < 2000].index

attack_df["Label"] = attack_df["Label"].replace(labels_to_replace, "Other")
sample_size = len(attack_df)

filtered_benign_df = benign_df.sample(n = sample_size , random_state = 42).reset_index (drop = True)

final_df = pd.concat([benign_df , attack_df] , axis= 0 ,ignore_index= True)

final_df = final_df.sample(frac = 1.0 ).reset_index (drop = True)


features = final_df.drop("Label" , axis=1).columns

label = "Label"

X = final_df[features]
y = final_df[label]

X_train , X_test, y_train , y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify= y,
    random_state=42
)


train_df = pd.concat([X_train , y_train] , axis =1 )
test_df = pd.concat([X_test , y_test] , axis =1 )

os.makedirs("./Data/filtered_data", exist_ok= True)

train_df.to_parquet("./filtered_data/train.parquet")
test_df.to_parquet("./filtered_data/test.parquet")
