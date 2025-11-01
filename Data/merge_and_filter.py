import pandas as pd 
import glob 
import os 

import seaborn as sns
import matplotlib.pyplot as plt
folder = "./Data/raw_data/"
search_pattern = os.path.join(folder , "*.parquet")
all_files = glob.glob(search_pattern)

df_list = [pd.read_parquet(file) for file in all_files]
df = pd.concat(df_list , axis=0 , ignore_index= True)
#selecting ftp-patator as the class whose data will be hidden while training
test_label = "FTP-Patator"

test_df = df[df["Label"] == test_label].copy()
imbalanced_train_df = df[df["Label"] != test_label].copy()
del df

#transformin the problem to a binary classification problem by classifying the data as benign or attack
imbalanced_train_df["Label"] = (imbalanced_train_df["Label"] != "Benign").astype(int)
test_df["Label"] = 1

#the dataset is highly imbalaced 
label_counts = imbalanced_train_df['Label'].value_counts()
label_map = {0: "Benign", 1: "Attack"}
color_map = {"Benign": "blue", "Attack": "red"} 
label_counts.index = label_counts.index.map(label_map)

ax = sns.barplot(x=label_counts.index, y=label_counts.values , hue = label_counts.values)
plt.title("Training Set Class Balance")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

#sampling from the benign class to deal with class imbalance
train_df_benign = imbalanced_train_df[imbalanced_train_df["Label"] == 0].copy()
train_df_attack = imbalanced_train_df[imbalanced_train_df["Label"] == 1].copy()

sample_size = len(train_df_attack)
train_df_benign = train_df_benign.sample(n = sample_size ,random_state= 42)
filtered_train_df = pd.concat([train_df_attack , train_df_benign] , axis = 0 , ignore_index= True)

#shuffling the data
filtered_train_df = filtered_train_df.sample(frac= 1.0 ,random_state=42).reset_index(drop= True)

# the train dataset is balanced now
label_counts = filtered_train_df['Label'].value_counts()
label_map = {0: "Benign", 1: "Attack"}
color_map = {"Benign": "blue", "Attack": "red"} 
label_counts.index = label_counts.index.map(label_map)

ax = sns.barplot(x=label_counts.index, y=label_counts.values , hue= label_counts.values)
plt.title("Training Set Class Balance")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()


new_data_percentage = len(filtered_train_df)/len(imbalanced_train_df) * 100

print(f"Filtered train dataset size :{len(filtered_train_df)}\nThe new dataset is {new_data_percentage}% of the original")

os.makedirs("./Data/filtered_data/", exist_ok= True)

# #saving the data
filtered_train_df.to_parquet("./Data/filtered_data/train_data.parquet" )
test_df.to_parquet("./Data/filtered_data/test_ftp-patator.parquet" )

