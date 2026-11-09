"""Freeze temporal split v1: first 80% train, last 20% test, by row order."""
import pandas as pd

df = pd.read_csv("data/bank-additional-full.csv", sep=";")
n_train = int(len(df) * 0.8)

train_idx = pd.DataFrame({"row_id": df.index[:n_train]})
test_idx = pd.DataFrame({"row_id": df.index[n_train:]})

train_idx.to_csv("data/splits/v1/train_index.csv", index=False)
test_idx.to_csv("data/splits/v1/test_index.csv", index=False)

print(f"Total rows: {len(df)}")
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
