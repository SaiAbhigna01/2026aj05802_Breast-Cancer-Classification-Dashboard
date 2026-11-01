
# ==========================================================
# MACHINE LEARNING ASSIGNMENT
# Step 3: Training Five Classification Models
# Dataset: Breast Cancer Wisconsin Diagnostic
# ==========================================================

import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ----------------------------------------------------------
# 1. Load training dataset
# ----------------------------------------------------------

train_data = pd.read_csv("train_data.csv")

# Separate features and target

X_train = train_data.drop(columns=["Diagnosis"])
y_train = train_data["Diagnosis"]

print("TRAINING DATA LOADED")
print("Training records:", X_train.shape[0])
print("Number of features:", X_train.shape[1])

# ----------------------------------------------------------
# 2. Create model directory
# ----------------------------------------------------------

os.makedirs("model", exist_ok=True)

# ----------------------------------------------------------
# 3. Define five classification models
# ----------------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}

# ----------------------------------------------------------
# 4. Train and save all five models
# ----------------------------------------------------------

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

print("\n========== MODEL TRAINING ==========")

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Train model using training data
    model.fit(X_train, y_train)

    # Save trained model
    filename = model_files[model_name]

    joblib.dump(model, os.path.join("model", filename))

    print(f"{model_name} trained successfully!")
    print(f"Saved as: model/{filename}")

# ----------------------------------------------------------
# 5. Training completion
# ----------------------------------------------------------

print("\n======================================")
print("ALL FIVE MODELS TRAINED SUCCESSFULLY!")
print("======================================")

print("\nSaved model files:")

for filename in model_files.values():
    print("model/" + filename)

print("\nMODEL TRAINING COMPLETED!")
