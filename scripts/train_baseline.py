"""Week 1 baseline: logistic regression on the frozen temporal split.

Reads data/splits/v1/{train,test}_index.csv from Session B.
Drops `duration` (leakage, see docs/data_card.md). No tuning.
Prints AUC, log-loss, and Brier score on the test split.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

print(f"Train rows: {len(train)}, test rows: {len(test)}")
print(f"Test positive rate: {y_test.mean():.1%}")
print(f"AUC:      {roc_auc_score(y_test, proba):.3f}")
print(f"Log-loss: {log_loss(y_test, proba):.3f}")
print(f"Brier:    {brier_score_loss(y_test, proba):.3f}")

