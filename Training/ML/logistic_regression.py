import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score , f1_score

train_val_df = pd.read_parquet("./Data/filtered_data/train_data.parquet")
test_df = pd.read_parquet("/Data/filtered_data/test_ftp-patator.parquet")

features = [c for c in train_val_df.drop(["Label"] , axis = 1).columns ]
label = "Label"
X_train_val = train_val_df[features]
y_train_val = train_val_df[label]

X_test = test_df[features]
y_test = test_df[label]

kf = StratifiedKFold(n_splits= 5)

acc = []
prec= []
recall = []
f1 = []
for  train_idx , val_idx in kf.split(X= X_train_val , y=y_train_val):
    X_train ,X_val = X_train_val.iloc[train_idx].copy() , X_train_val.iloc[val_idx].copy()
    y_train ,y_val = y_train_val.iloc[train_idx].copy() , y_train_val.iloc[val_idx].copy()
    
    imputer = SimpleImputer(strategy= "median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    
    lr= LogisticRegression(max_iter = 1000)
    lr.fit(X_train, y_train)    

    y_val_preds = lr.predict(X_val)
    fold_acc = accuracy_score(y_true = y_val , y_pred=y_val_preds)
    fold_prec = precision_score(y_true = y_val , y_pred=y_val_preds)
    fold_recall = recall_score(y_true = y_val , y_pred=y_val_preds)
    fold_f1 = f1_score(y_true = y_val , y_pred=y_val_preds)
    acc.append(fold_acc)
    prec.append(fold_prec)
    recall.append(fold_recall)
    f1.append(fold_f1)
    
avg_acc = np.mean(acc)
avg_prec = np.mean(prec)
avg_recall = np.mean(recall)
avg_f1 = np.mean(f1)


print(f"average val accuracy :{avg_acc} , average val prec :{avg_prec}, average val recall :{avg_recall} , average val f1:{avg_f1}")

imputer = SimpleImputer(strategy= "median")
X_train_val = imputer.fit_transform(X_train_val)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.fit_transform(X_test)

lr = LogisticRegression(max_iter= 1000)
lr.fit(X_train_val , y_train_val)

y_test_preds = lr.predict(X_test)

test_acc = accuracy_score(y_true = y_test , y_pred=y_test_preds)
test_prec = precision_score(y_true = y_test , y_pred=y_test_preds)
test_recall = recall_score(y_true = y_test , y_pred=y_test_preds)
test_f1 = f1_score(y_true = y_test , y_pred=y_test_preds)

print(f"test val accuracy :{test_acc} , test val prec :{test_prec}, test val recall :{test_recall} , test val f1:{test_f1}")
