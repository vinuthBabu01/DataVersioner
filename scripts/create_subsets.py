import os
import pandas as pd
from sklearn.datasets import load_iris

# Ensure v1 and v2 folders exist
os.makedirs("data/v1", exist_ok=True)
os.makedirs("data/v2", exist_ok=True)

# Load iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# v1 = first 100 rows
df.iloc[:100].to_csv("data/v1/dataset.csv", index=False)

# v2 = last 50 rows
df.iloc[100:].to_csv("data/v2/dataset.csv", index=False)

print("✅ Created v1 (100 rows) and v2 (50 rows) subsets of Iris dataset")
