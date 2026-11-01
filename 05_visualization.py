
# =====================================================
# MACHINE LEARNING ASSIGNMENT
# Step 5: Model Comparison and Visualization
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import confusion_matrix

# 1. Create results folder
os.makedirs("results", exist_ok=True)

# 2. Load comparison results
comparison = pd.read_csv("model_comparison.csv")

print("MODEL COMPARISON TABLE")
print(comparison.round(4).to_string(index=False))

# 3. Plot comparison of evaluation metrics

metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

comparison_plot = comparison.set_index("ML Model")[metrics]

plt.figure(figsize=(13, 7))

comparison_plot.plot(
    kind="bar",
    figsize=(13, 7),
    width=0.8
)

plt.title("Performance Comparison of Five ML Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Metric Score")
plt.xticks(rotation=20)
plt.ylim(-1.05, 1.05)
plt.legend(title="Evaluation Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("results/model_comparison.png", dpi=300)
plt.show()

print("\nComparison graph saved!")

# 4. Load test data
test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop(columns=["Diagnosis"])
y_test = test_data["Diagnosis"]

# 5. Define saved model paths
model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

# 6. Generate confusion matrix for each model

for model_name, file_path in model_files.items():

    model = joblib.load(file_path)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join("results", filename),
        dpi=300
    )

    plt.show()
    plt.close()

    print(f"Saved confusion matrix: {filename}")

print("\nALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
