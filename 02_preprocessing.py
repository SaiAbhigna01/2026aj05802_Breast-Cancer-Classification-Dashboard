
# =====================================================
# MACHINE LEARNING ASSIGNMENT
# Step 2: Data Preprocessing
# Dataset: Breast Cancer Wisconsin Diagnostic
# =====================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the dataset
df = pd.read_csv("breast_cancer_data.csv")

print("DATASET LOADED SUCCESSFULLY")
print("Original dataset shape:", df.shape)

# 2. Separate input features and target

X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

print("\nInput feature shape:", X.shape)
print("Target shape:", y.shape)

# 3. Convert target labels into numerical values
# B = 0 (Benign)
# M = 1 (Malignant)

y = y.map({"B": 0, "M": 1})

print("\nTarget values after encoding:")
print(y.value_counts())

# 4. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Display split information

print("\n========== DATA SPLITTING RESULTS ==========")

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

print("Training features:", X_train.shape[1])
print("Testing features:", X_test.shape[1])

# 6. Check class distribution

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# 7. Combine features and target for saving

train_data = X_train.copy()
train_data["Diagnosis"] = y_train

test_data = X_test.copy()
test_data["Diagnosis"] = y_test

# 8. Save training and testing datasets

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("\n========== FILES SAVED ==========")
print("Training data saved as train_data.csv")
print("Testing data saved as test_data.csv")

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")

