import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_val_df = pd.read_parquet("../..//Data/filtered_data/train_data.parquet")
test_df = pd.read_parquet("../../Data/filtered_data/test_ftp-patator.parquet")

train_val_df = train_val_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

features = [c for c in train_val_df.drop(["Label"], axis=1).columns]
label = "Label"

X_train_val = train_val_df[features]
y_train_val = train_val_df[label]
X_test = test_df[features]
y_test = test_df[label]

kf = StratifiedKFold(n_splits=5)

acc, prec, recall, f1 = [], [], [], []
best_iterations = []

for train_idx, val_idx in kf.split(X=X_train_val, y=y_train_val):
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    lgbm = lgb.LGBMClassifier(device='gpu', n_estimators=1000, objective='binary')
    lgbm.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             eval_metric='binary_logloss',
             callbacks=[lgb.early_stopping(10, verbose=False)])    

    if lgbm.best_iteration_:
        best_iterations.append(lgbm.best_iteration_)
    else:
        best_iterations.append(1000)

    y_val_preds = lgbm.predict(X_val)
    fold_acc = accuracy_score(y_true=y_val, y_pred=y_val_preds)
    fold_prec = precision_score(y_true=y_val, y_pred=y_val_preds)
    fold_recall = recall_score(y_true=y_val, y_pred=y_val_preds)
    fold_f1 = f1_score(y_true=y_val, y_pred=y_val_preds)
    
    acc.append(fold_acc)
    prec.append(fold_prec)
    recall.append(fold_recall)
    f1.append(fold_f1)
    
avg_acc = np.mean(acc)
avg_prec = np.mean(prec)
avg_recall = np.mean(recall)
avg_f1 = np.mean(f1)
avg_best_iter = int(np.mean(best_iterations))

print(f"LGBM average val accuracy :{avg_acc:.4f} , LGBM average val prec :{avg_prec:.4f}, LGBM average val recall :{avg_recall:.4f} , LGBM average val f1:{avg_f1:.4f}")
print(f"Average best iteration from CV: {avg_best_iter}")

final_lgbm = lgb.LGBMClassifier(device='gpu', n_estimators=avg_best_iter, objective='binary')
final_lgbm.fit(X_train_val, y_train_val)

y_test_preds = final_lgbm.predict(X_test)

test_acc = accuracy_score(y_true=y_test, y_pred=y_test_preds)
test_prec = precision_score(y_true=y_test, y_pred=y_test_preds)
test_recall = recall_score(y_true=y_test, y_pred=y_test_preds)
test_f1 = f1_score(y_true=y_test, y_pred=y_test_preds)

print(f"LGBM Test accuracy :{test_acc:.4f} , LGBM Test prec :{test_prec:.4f}, LGBM Test recall :{test_recall:.4f} , LGBM Test f1:{test_f1:.4f}")

