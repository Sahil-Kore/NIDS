import pandas as pd 
from sklearn.metrics import accuracy_score , precision_score, recall_score , f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import numpy as np 

import os 

cpu_count = os.cpu_count()

train_val_df = pd.read_parquet("../../Data/filtered_data/train.parquet")
test_df = pd.read_parquet("../../Data/filtered_data/test.parquet")

features= train_val_df.drop("Label", axis =1 ).columns
label = "Label"


X_train_val = train_val_df[features]
X_test = test_df[features]
y_train_val = train_val_df[label]
y_test = test_df[label]

le = LabelEncoder()
y_train_val = le.fit_transform(y_train_val)
y_test = le.transform(y_test)

kf = StratifiedKFold()
acc = []
prec = []
recall = []
f1 = []
for train_idx ,val_idx in kf.split(X_train_val , y_train_val):
    print("Started")
    X_train , X_val = X_train_val.iloc[train_idx] ,X_train_val.iloc[val_idx]
    y_train , y_val = y_train_val[train_idx] , y_train_val[val_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    lr = LogisticRegression(max_iter = 1000 , n_jobs= cpu_count-2)
    lr.fit(X_train, y_train)
    
    predictions = lr.predict(X_val)
    fold_acc = accuracy_score(y_val , predictions)
    fold_prec = precision_score(y_val , predictions, average= "macro" , zero_division= 0) 
    fold_recall = recall_score(y_val , predictions , average = "macro", zero_division= 0) 
    fold_f1 = f1_score(y_val , predictions ,average= "macro" , zero_division= 0) 
    print("finished")
    acc.append(fold_acc)
    prec.append(fold_prec)
    recall.append(fold_recall)
    f1.append(fold_f1)

avg_acc = np.mean(acc)
avg_prec = np.mean(prec)
avg_recall = np.mean(recall)
avg_f1 = np.mean(f1)


print(f"Average accuracy : {avg_acc}")
print(f"Average precision : {avg_prec}")
print(f"Average recall : {avg_recall}")
print(f"Average f1 : {avg_f1}")


scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, n_jobs= cpu_count - 2)
lr.fit(X_train , y_train)
test_predictions = lr.predict(X_test)

test_acc = accuracy_score(y_test, test_predictions)
test_prec = precision_score(y_test, test_predictions, average="macro", zero_division=0)
test_recall = recall_score(y_test, test_predictions, average="macro", zero_division=0)
test_f1 = f1_score(y_test, test_predictions, average="macro", zero_division=0)



print("\n--- Saving Results to CSV ---")

model_name = "LogisticRegression"
results_csv_file = "../../model_evaluation_results.csv"

results_data = [
    {
        "model_name": model_name,
        "dataset": "val",  
        "acc": avg_acc,
        "pre": avg_prec,
        "recall": avg_recall,
        "f1": avg_f1
    },
    {
        "model_name": model_name,
        "dataset": "test", 
        "acc": test_acc,
        "pre": test_prec,
        "recall": test_recall,
        "f1": test_f1
    }
]

df_new_results = pd.DataFrame(results_data)

if os.path.exists(results_csv_file):
    df_new_results.to_csv(results_csv_file, mode='a', header=False, index=False)
    print(f"Appended results for '{model_name}' to {results_csv_file}")
else:
    df_new_results.to_csv(results_csv_file, mode='w', header=True, index=False)
    print(f"Created new results file: {results_csv_file}")