import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


train_val_df = pd.read_parquet("../../Data/train_val.parquet")
test_df = pd.read_parquet("../../Data/test.parquet")
features = [column for column in train_val_df.columns if column not in ["Label"]]
label = "Label"

X_train_val = train_val_df[features]
y_train_val = train_val_df[label]

X_test = test_df[features]
y_test = test_df[label]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []
acc_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X=X_train_val, y=y_train_val)):
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val= imputer.transform(X_val)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    lgb = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=-1,
        device="gpu",
        verbosity=-1,
        min_child_samples=200,
    )

    lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)],
    )

    y_pred = lgb.predict(X_val)

    f1 = f1_score(y_true=y_val, y_pred=y_pred, average="macro")
    acc = accuracy_score(y_true=y_val, y_pred=y_pred)

    f1_scores.append(f1)
    acc_scores.append(acc)

    print(f"fold :{fold} , accuracy{acc} , f1: {f1}")

avg_acc = np.mean(acc_scores)
avg_f1 = np.mean(f1_scores)
print(f"Average train accuracy :{avg_acc} , average train  f1 :{avg_f1}")

imputer = SimpleImputer(strategy= "median")
X_train_val = imputer.fit_transform(X_train_val)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

lgb = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=-1,
    device="gpu",
    verbosity=-1,
    min_child_samples=200,
)

lgb.fit(X_train_val, y_train_val)

y_test_preds = lgb.predict(X_test )

test_acc =accuracy_score(y_test , y_test_preds)
test_f1 = f1_score(y_test , y_test_preds)

print(f"Test accuracy :{test_acc} , test f1:{test_f1}")

