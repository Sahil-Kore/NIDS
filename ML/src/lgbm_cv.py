import sys
import os

from lightgbm import LGBMClassifier, early_stopping , log_evaluation

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

grandparent_dir = os.path.dirname(parent_dir)

if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
    
from HelperModule.helper_functions import get_df  

df = get_df(sample_frac=1.0)

features = [column for column in df.columns if column not in ["Label"]]
label = "Label"

lbe = LabelEncoder()
lbe.fit(df[label])
df[label] = lbe.transform(df[label])

X = df[features]
y = df[label]


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []
acc_scores = []
for train_idx, val_idx in kf.split(X=X, y=y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_mean = X_train.mean()
    X_train = X_train.fillna(train_mean)
    X_val = X_val.fillna(train_mean)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    lgb = LGBMClassifier(
        n_estimators=1000, 
        learning_rate=0.01,
        max_depth=-1,
        device="gpu",
        verbosity=-1,
        objective= "multiclass",
        min_child_samples=200
    )

    lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)],
    )
    
    y_pred = lgb.predict(X_val)

    f1 = f1_score(y_true = y_val , y_pred= y_pred , average="macro")
    acc = accuracy_score(y_true = y_val , y_pred=y_pred)

    f1_scores.append(f1)
    acc_scores.append(acc)


roc_score = np.mean(f1_scores)
acc_score = np.mean(acc_scores)

print(f"macro f1 score {roc_score}\n accuracy score {acc_score}")
