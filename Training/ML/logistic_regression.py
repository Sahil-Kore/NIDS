import numpy as np
import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score , f1_score
from sklearn.pipeline import Pipeline

train_val_df = pd.read_parquet("./Data/filtered_data/train_data.parquet")
test_df = pd.read_parquet("./Data/filtered_data/test_ftp-patator.parquet")

train_val_df = train_val_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

features = [c for c in train_val_df.drop(["Label"] , axis = 1).columns ]
label = "Label"
X_train_val = train_val_df[features]
y_train_val = train_val_df[label]

X_test = test_df[features]
y_test = test_df[label]

kf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)

acc = []
prec= []
recall = []
f1 = []

print("Starting  cross-validation...")

for  fold,(train_idx , val_idx) in enumerate(kf.split(X= X_train_val , y=y_train_val)):
    X_train ,X_val = X_train_val.iloc[train_idx] , X_train_val.iloc[val_idx]
    y_train ,y_val = y_train_val.iloc[train_idx] , y_train_val.iloc[val_idx]
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)    

    y_val_preds = pipeline.predict(X_val)
    
    fold_acc = accuracy_score(y_true = y_val , y_pred=y_val_preds)
    fold_prec = precision_score(y_true = y_val , y_pred=y_val_preds)
    fold_recall = recall_score(y_true = y_val , y_pred=y_val_preds)
    fold_f1 = f1_score(y_true = y_val , y_pred=y_val_preds)
    
    acc.append(fold_acc)
    prec.append(fold_prec)
    recall.append(fold_recall)
    f1.append(fold_f1)
    print(f"Fold:{fold}, val accuracy :{fold_acc:.4f} , fold val prec :{fold_prec:.4f}, fold val recall :{fold_recall:.4f} , fold val f1:{fold_f1:.4f}")
    
avg_acc = np.mean(acc)
avg_prec = np.mean(prec)
avg_recall = np.mean(recall)
avg_f1 = np.mean(f1)

print(f"Average val accuracy :{avg_acc:.4f} , Average val prec :{avg_prec:.4f}, Average val recall :{avg_recall:.4f} , Average val f1:{avg_f1:.4f}")

print("Training final training")

final_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

final_pipeline.fit(X_train_val , y_train_val)

y_test_preds = final_pipeline.predict(X_test)

test_acc = accuracy_score(y_true = y_test , y_pred=y_test_preds)
test_prec = precision_score(y_true = y_test , y_pred=y_test_preds)
test_recall = recall_score(y_true = y_test , y_pred=y_test_preds)
test_f1 = f1_score(y_true = y_test , y_pred=y_test_preds)

print(f"Test accuracy :{test_acc:.4f} , Test prec :{test_prec:.4f}, Test recall :{test_recall:.4f} , Test f1:{test_f1:.4f}")

pipeline_dir = "./Training/ML/models/"
os.makedirs(pipeline_dir, exist_ok=True)
pipeline_path = os.path.join(pipeline_dir, "lr_pipeline.joblib")

print(f"Saving pipeline to {pipeline_path}...")
joblib.dump(final_pipeline, pipeline_path)

print("Pipeline saved.")

