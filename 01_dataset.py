
# =====================================================
# MACHINE LEARNING ASSIGNMENT
# Step 1: Fetch and Explore the UCI Dataset
# Dataset: Breast Cancer Wisconsin Diagnostic
# =====================================================

import pandas as pd
from ucimlrepo import fetch_ucirepo

# 1. Fetch the dataset from UCI repository
print("Downloading dataset...")

breast_cancer = fetch_ucirepo(id=17)

# 2. Extract input features (X)
X = breast_cancer.data.features

# 3. Extract target variable (y)
y = breast_cancer.data.targets

# 4. Display dataset metadata
print("\n========== DATASET METADATA ==========")
print(breast_cancer.metadata)

# 5. Display variable information
print("\n========== VARIABLE INFORMATION ==========")
print(breast_cancer.variables)

# 6. Display first five input records
print("\n========== INPUT FEATURES ==========")
print(X.head())

# 7. Display target values
print("\n========== TARGET VALUES ==========")
print(y.head())

# 8. Display dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Number of records:", X.shape[0])
print("Number of features:", X.shape[1])

# 9. Combine input features and target
df = pd.concat(
    [X.reset_index(drop=True), y.reset_index(drop=True)],
    axis=1
)

# 10. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 11. Check target distribution
print("\n========== CLASS DISTRIBUTION ==========")
print(y.value_counts())

# 12. Save dataset as CSV
df.to_csv("breast_cancer_data.csv", index=False)

print("\nDataset successfully saved as breast_cancer_data.csv")

# 13. Display final dataset information
print("\n========== FINAL DATASET ==========")
print(df.shape)
print(df.head())
