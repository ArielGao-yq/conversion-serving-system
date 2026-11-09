"""Sanity-check the downloaded dataset and print the duration leakage evidence."""
import pandas as pd

df = pd.read_csv("data/bank-additional-full.csv", sep=";")

print("Shape:", df.shape)
print("\nTarget counts:")
print(df["y"].value_counts())
print("\nPositive rate:", round((df["y"] == "yes").mean(), 4))

zero = df[df["duration"] == 0]
print("\nRows with duration == 0:", len(zero))
print("Their y values:", zero["y"].unique())
print("\nMean duration by outcome:")
print(df.groupby("y")["duration"].mean())
