
# ==========================================================
# MACHINE LEARNING ASSIGNMENT
# Step 4: Evaluation of Five Classification Models
# ==========================================================

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ----------------------------------------------------------
# 1. Load test dataset
# ----------------------------------------------------------

test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop(columns=["Diagnosis"])
y_test = test_data["Diagnosis"]

print("TEST DATA LOADED SUCCESSFULLY")
print("Test records:", X_test.shape[0])

# ----------------------------------------------------------
# 2. Define saved model paths
# ----------------------------------------------------------

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

# ----------------------------------------------------------
# 3. Create results list
# ----------------------------------------------------------

results = []

# ----------------------------------------------------------
# 4. Evaluate each model
# ----------------------------------------------------------

for model_name, file_path in model_files.items():

    print("\n====================================")
    print("Evaluating:", model_name)
    print("====================================")

    # Load saved model
    model = joblib.load(file_path)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Generate prediction probabilities for AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Store results
    results.append({
        "ML Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Display individual results
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")

    # Display confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"],
        zero_division=0
    ))

# ----------------------------------------------------------
# 5. Create comparison table
# ----------------------------------------------------------

comparison = pd.DataFrame(results)

print("\n============================================")
print("FINAL MODEL COMPARISON")
print("============================================")

print(comparison.round(4).to_string(index=False))

# ----------------------------------------------------------
# 6. Save comparison table
# ----------------------------------------------------------

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nComparison table saved as model_comparison.csv")

# ----------------------------------------------------------
# 7. Identify the best model using F1 Score
# ----------------------------------------------------------

best_model = comparison.loc[
    comparison["F1 Score"].idxmax()
]

print("\n============================================")
print("OVERALL WINNER BASED ON F1 SCORE")
print("============================================")

print("Best Model:", best_model["ML Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))

print("\nMODEL EVALUATION COMPLETED SUCCESSFULLY!")
