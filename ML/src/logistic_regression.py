import pandas as pd 
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import helper_functions
from sklearn .preprocessing import RobustScaler



df= helper_functions.get_df(sample_frac = 1.0)

features = [column for column in df.columns if column not in ["Label"]]
label = "Label"

df[label] =(df[label] != "Benign") .astype(int) 

X= df[features]
y= df[label]


X= df[features]
kf = StratifiedKFold(n_splits= 5, random_state= 42 , shuffle= True)

for train_idx, test_idx in kf.split(X=X,y=y):
    X_train , X_test = X.iloc[train_idx] , X.iloc[test_idx]
    y_train , y_test = y.iloc[train_idx] , y.iloc[test_idx]


lr = LogisticRegression()

lr.fit (X_train , y_train)

y_preds = lr.predict(X_test)


score = roc_auc_score(y_true=y_test , y_score= y_preds)
print(score)
acc = accuracy_score(y_true= y_test , y_pred = y_preds)
print(acc)