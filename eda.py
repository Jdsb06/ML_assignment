import pandas as pd
import numpy as np

train1 = pd.read_csv('IMT2024022/IMT2024022_train_var1.csv')
test1 = pd.read_csv('IMT2024022/IMT2024022_test_var1.csv')
train2 = pd.read_csv('IMT2024022/IMT2024022_train_var2.csv')
test2 = pd.read_csv('IMT2024022/IMT2024022_test_var2.csv')

print("=== VAR 1 ===")
print("Train shape:", train1.shape)
print("Test shape:", test1.shape)
print("Missing values train:", train1.isnull().sum().to_dict())
print("Missing values test:", test1.isnull().sum().to_dict())
print("Train description:\n", train1.describe().T[['mean', 'std', 'min', '50%', 'max']])
print("Test description:\n", test1.describe().T[['mean', 'std', 'min', '50%', 'max']])

print("\n=== VAR 2 ===")
print("Train shape:", train2.shape)
print("Test shape:", test2.shape)
print("Missing values train:", train2.isnull().sum().to_dict())
print("Missing values test:", test2.isnull().sum().to_dict())
print("Train description:\n", train2.describe().T[['mean', 'std', 'min', '50%', 'max']])
print("Test description:\n", test2.describe().T[['mean', 'std', 'min', '50%', 'max']])
