import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

from tabpfn import TabPFNClassifier


df = pd.read_parquet("../Data/train_val.parquet")
sub_df_list = []
for label, group_df in df.groupby("Label"):
    sub_df = group_df.sample(n=500, random_state=42)
    sub_df_list.append(sub_df)

final_df = pd.concat(sub_df_list, axis=0, ignore_index=True)

features = [c for c in final_df.columns if c not in ["Label"]]
label = "Label"

X_train_val = final_df[features]
y_train_val = final_df[label]


kf = StratifiedKFold(n_splits=5)
acc = []
f1 = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X=X_train_val, y=y_train_val)):
    X_train, y_train = (
        X_train_val.iloc[train_idx].copy(),
        y_train_val.iloc[train_idx].copy(),
    )
    X_val, y_val = X_train_val.iloc[val_idx].copy(), y_train_val.iloc[val_idx].copy()

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    classifier = TabPFNClassifier(device="cuda")
    classifier.fit(X_train, y_train)
    y_val_preds = classifier.predict(X_val)

    fold_acc = accuracy_score(y_true=y_val, y_pred=y_val_preds)
    fold_f1 = f1_score(y_true=y_val, y_pred=y_val_preds)

    acc.append(fold_acc)
    f1.append(fold_f1)

    print(f"fold:{fold} , accuracy :{fold_acc} , f1:{fold_f1}")

avg_acc = np.mean(acc)
avg_f1 = np.mean(f1)
print(f"Average accuracy :{avg_acc} , Average f1 :{avg_f1}")

test_df = pd.read_parquet("../Data/test.parquet")
X_test = test_df[features]
y_test = test_df[label]

imputer = SimpleImputer(strategy="median")
X_train_val = imputer.fit_transform(X_train_val)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

classifier = TabPFNClassifier(device="cuda")
classifier.fit(X_train_val, y_train_val)

y_test_preds = classifier.predict(X_test)

test_acc = accuracy_score(y_true=y_test, y_pred=y_test_preds)
test_f1 = f1_score(y_true=y_test, y_pred=y_test_preds)

print(f"Test accuracy :{test_acc} , test f1 :{test_f1}")

print(y_test_preds)