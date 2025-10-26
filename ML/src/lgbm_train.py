import sys
import os

import pandas as pd 
import numpy as np

from lightgbm import LGBMClassifier, early_stopping , log_evaluation

from sklearn.model_selection import train_test_split # Use train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

from HelperModule.helper_functions import get_df

df = get_df(sample_frac=1.0) 

print("Cleaning data...")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
print("Cleaning complete.")

features = [column for column in df.columns if column not in ["Label"]]
label = "Label"

lbe = LabelEncoder()
df[label] = lbe.fit_transform(df[label]) 

X = df[features]
y = df[label]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

imputer = SimpleImputer(strategy= "median")

scaler = StandardScaler()

# --- Model Training ---
lgb = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=-1,
    device="gpu",
    verbosity=-1,
    objective="multiclass",
    min_child_samples=200, 
    random_state=42 
)
pipeline = Pipeline([
    ('imputer' , imputer),
    ("scaler" , scaler),
    ("classifier" , lgb)
])

print("Training Pipeline")

pipeline.fit(X_train, y_train)

print("\nEvaluating on validation set...")
y_pred = pipeline.predict(X_val) 

# Calculate scores
f1 = f1_score(y_true=y_val, y_pred=y_pred, average="macro")
acc = accuracy_score(y_true=y_val, y_pred=y_pred)

print(f"\nValidation Macro F1 Score: {f1:.4f}")
print(f"Validation Accuracy Score: {acc:.4f}")

import joblib
joblib.dump(pipeline , "../models/lgb_1.0.joblib")

print("Pipeline saved")