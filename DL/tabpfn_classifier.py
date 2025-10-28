import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from tabpfn import TabPFNClassifier
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)


if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
    
from  HelperModule.helper_functions import get_df

df = get_df(sample_frac= 1.0)

sub_df_list = []
for label , group_df in df.groupby("Label"):
    sub_df = group_df.sample(n=100 , random_state=42)
    sub_df_list.append(sub_df)

final_df = pd.concat(sub_df_list , axis = 0 , ignore_index= True)

features = [c for c in final_df.columns if c not in ["Label"]]
label  = "Label"

X= final_df[features]
y = final_df[label]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train ,X_val , y_train , y_val = train_test_split(
    X_scaled, y,
    test_size= 0.2 , 
    random_state= 42, 
    stratify= y
)

classifier = TabPFNClassifier(device = "cuda" )

print("Starting fitting ")
classifier.fit(X_train , y_train)

print("Predicting")
y_pred = classifier.predict(X_val)

accuracy = accuracy_score(y_val , y_pred)
f1 = f1_score(y_val , y_pred , average= "macro")

print(f"accuracy ={accuracy}")
print(f"macro f1 {f1}")