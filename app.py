
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# WEBSITE INTRODUCTION
# ==========================================
st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="🎗️",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #fffaff;
}

h1 {
    color: #70245f;
    font-size: 38px !important;
}

h2, h3 {
    color: #8e347d;
}

.info-box {
    background: linear-gradient(120deg, #fff0f8, #f5edff);
    padding: 22px;
    border-radius: 15px;
    border-left: 6px solid #d65ca8;
    margin: 15px 0;
}

.feature-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #ead3e8;
    min-height: 175px;
    box-shadow: 0 3px 10px rgba(100, 40, 100, 0.07);
}
</style>
""", unsafe_allow_html=True)

# Main heading
st.title("🎗️ Breast Cancer Classification Dashboard")

st.subheader("Machine Learning Model Evaluation")

# About section
st.markdown("""
<div class="info-box">
<h3>About This Application</h3>

This interactive dashboard demonstrates a Machine Learning
classification system designed to classify breast cancer
diagnosis as <b>Benign</b> or <b>Malignant</b>.

It uses the Breast Cancer Wisconsin (Diagnostic) dataset
obtained from the UCI Machine Learning Repository.

The application allows users to upload test data, select
a trained classification model, and evaluate its performance.
</div>
""", unsafe_allow_html=True)

# Project information cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
    <h3>📊 Dataset</h3>
    <p>569 records and 30 numerical features from the
    Breast Cancer Wisconsin Diagnostic dataset.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
    <h3>🤖 ML Models</h3>
    <p>Five classification algorithms:
    Logistic Regression, Decision Tree, KNN,
    Gaussian Naive Bayes and SVM.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
    <h3>📈 Evaluation</h3>
    <p>Model performance is measured using Accuracy,
    AUC, Precision, Recall, F1-score and MCC,
    along with a Confusion Matrix.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
### How to Use This Dashboard

1. Upload a test dataset in CSV format containing the
   actual diagnosis column.
2. Select one of the five trained machine learning models.
3. View the evaluation metrics to understand model performance.
4. Examine the confusion matrix and classification report.
5. Compare models to identify the best-performing classifier.

**Objective:** To compare multiple machine learning
classification algorithms and understand their effectiveness
in distinguishing benign and malignant breast tumor cases.
""")

st.markdown("---")
st.caption("Machine Learning Classification Assignment | Breast Cancer Wisconsin Diagnostic Dataset")

st.success("Welcome! Your Streamlit application is running.")

st.header("About the Project")

st.write("""
Dataset: Breast Cancer Wisconsin Diagnostic

Number of records: 569

Number of input features: 30

Classification: Benign and Malignant

Machine Learning Models:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine
""")

# ==========================================
# MODEL FILE PATHS
# ==========================================

model_paths = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

# ==========================================
# CSV DATASET UPLOAD
# ==========================================

st.divider()

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose your test_data.csv file",
    type=["csv"]
)

# ==========================================
# MODEL SELECTION
# ==========================================

st.divider()

st.header("Select Machine Learning Model")

selected_model = st.selectbox(
    "Choose your classification model:",
    list(model_paths.keys())
)

st.success(f"You selected: {selected_model}")

# ==========================================
# MODEL EVALUATION
# ==========================================

if uploaded_file is not None:

    try:

        # Read uploaded dataset
        test_data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # Check target column
        if "Diagnosis" not in test_data.columns:

            st.error("Diagnosis column is missing!")

            st.stop()

        # Display dataset preview
        st.subheader("Dataset Preview")

        st.dataframe(test_data.head())

        st.write("Total records:", test_data.shape[0])
        st.write("Total columns:", test_data.shape[1])

        # Separate features and target
        X = test_data.drop(columns=["Diagnosis"])
        y = test_data["Diagnosis"]

        # Check target values
        if set(y.unique()) != {0, 1}:

            st.error("Diagnosis must contain encoded values 0 and 1.")

            st.stop()

        # Load selected trained model
        model = joblib.load(model_paths[selected_model])

        # Match training feature order
        X = X[model.feature_names_in_]

        # Generate predictions
        y_pred = model.predict(X)

        # Generate prediction probabilities
        y_prob = model.predict_proba(X)[:, 1]

        # ==========================================
        # EVALUATION METRICS
        # ==========================================

        st.divider()

        st.header("Model Evaluation Results")

        accuracy = accuracy_score(y, y_pred)

        auc = roc_auc_score(y, y_prob)

        precision = precision_score(
            y, y_pred, zero_division=0
        )

        recall = recall_score(
            y, y_pred, zero_division=0
        )

        f1 = f1_score(
            y, y_pred, zero_division=0
        )

        mcc = matthews_corrcoef(y, y_pred)

        # Display metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        # ==========================================
        # CONFUSION MATRIX
        # ==========================================

        st.divider()

        st.header("Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            linewidths=0.5,
            ax=ax
        )

        ax.set_xlabel("Predicted Diagnosis")
        ax.set_ylabel("Actual Diagnosis")

        ax.set_title(
            f"Confusion Matrix - {selected_model}"
        )

        st.pyplot(fig)

        plt.close(fig)

        # ==========================================
        # CLASSIFICATION REPORT
        # ==========================================

        st.divider()

        st.header("Classification Report")

        report = classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        st.success("Model evaluation completed successfully!")

    except Exception as e:

        st.error(f"An error occurred: {e}")

else:

    st.info("Please upload your test_data.csv file to view evaluation results.")
