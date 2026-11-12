"""Week 1 baseline: logistic regression on the frozen temporal split.

Reads data/splits/v1/{train,test}_index.csv from Session B.
Drops `duration` (leakage, see docs/data_card.md). No tuning.
Prints AUC, log-loss, and Brier score on the test split.
"""
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from src.features import build_preprocessor

df = pd.read_csv("data/bank-additional-full.csv", sep=";")
train_idx = pd.read_csv("data/splits/v1/train_index.csv")["row_id"]
test_idx = pd.read_csv("data/splits/v1/test_index.csv")["row_id"]

train = df.loc[train_idx]
test = df.loc[test_idx]

FEATURES = [c for c in df.columns if c not in ("y", "duration")]
X_train, y_train = train[FEATURES], (train["y"] == "yes").astype(int)
X_test, y_test = test[FEATURES], (test["y"] == "yes").astype(int)

numeric = X_train.select_dtypes(include="number").columns.tolist()
categorical = [c for c in FEATURES if c not in numeric]

model = Pipeline([
    ("prep", build_preprocessor(numeric, categorical)),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

print(f"Train rows: {len(train)}, test rows: {len(test)}")
print(f"Test positive rate: {y_test.mean():.1%}")
auc = roc_auc_score(y_test, proba)
logloss = log_loss(y_test, proba)
brier = brier_score_loss(y_test, proba)

mlflow.set_experiment('conversion-serving-system')
with mlflow.start_run(run_name='baseline_logreg_v1'):
    mlflow.log_param('model_type','logistic_regression')
    mlflow.log_param('split_version','v1')
    mlflow.log_param('max_iter',1000)
    mlflow.log_metric('auc',auc)
    mlflow.log_metric('log_loss',logloss)
    mlflow.log_metric('brier',brier)

print(f"AUC:      {auc:.3f}")
print(f"Log-loss: {logloss:.3f}")
print(f"Brier:    {brier:.3f}")
