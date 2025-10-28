import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_val_df = pd.read_parquet("../../Data/train_val.parquet")
test_df = pd.read_parquet("../../Data/test.parquet")

features = [c for c in train_val_df.columns if c not in ["Label"]]
label = "Label"

X_train_val = train_val_df[features].copy()
y_train_val = train_val_df[label].copy()

X_test = test_df[features].copy()
y_test = test_df[label].copy()

del train_val_df
del test_df

imputer = SimpleImputer(strategy="median")

X_train_val[features] = imputer.fit_transform(X_train_val[features])
X_test[features] = imputer.transform(X_test[features])

scaler = StandardScaler()
X_train_val[features] = scaler.fit_transform(X_train_val[features])
X_test[features] = scaler.transform(X_test[features])

kf = StratifiedKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(kf.split(X=X_train_val, y=y_train_val)):
    X_train, y_train = X_train_val.iloc[train_idx], y_train_val.iloc[train_idx]
    X_val, y_val = X_train_val.iloc[val_idx], y_train_val.iloc[val_idx]

    lr = LogisticRegression(random_state=42, max_iter=200  , C=10.0)
    lr.fit(X_train, y_train)
    y_val_preds = lr.predict(X_val)

    accuracy = accuracy_score(y_true=y_val, y_pred=y_val_preds)
    precision = precision_score(y_true=y_val, y_pred=y_val_preds)
    recall = recall_score(y_true=y_val, y_pred=y_val_preds)
    f1 = f1_score(y_true=y_val, y_pred=y_val_preds)

    print(
        f"fold:{fold} , accuracy:{accuracy} , precision:{precision} , recall:{recall} , f1:{f1}"
    )


# evaluating the model on test set
lr = LogisticRegression(random_state=42 , max_iter = 200 , C= 10.0)

lr.fit(X_train_val, y_train_val)
y_test_preds = lr.predict(X_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_test_preds)
precision = precision_score(y_true=y_test, y_pred=y_test_preds)
recall = recall_score(y_true=y_test, y_pred=y_test_preds)
f1 = f1_score(y_true=y_test, y_pred=y_test_preds)

print(
    f"Test data metrics \n accuracy:{accuracy} , precision:{precision} , recall:{recall} , f1:{f1}"
)
