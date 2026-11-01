
# Breast Cancer Classification Dashboard

## A. Problem Statement

The objective of this project is to develop and compare five Machine Learning classification algorithms to predict whether a breast tumour is Benign or Malignant.

The project uses the Breast Cancer Wisconsin Diagnostic dataset and evaluates the performance of five classification models using multiple evaluation metrics.

## B. Dataset Description

- Dataset Name: Breast Cancer Wisconsin Diagnostic
- Dataset Source: UCI Machine Learning Repository
- Total Instances: 569
- Number of Input Features: 30
- Target Variable: Diagnosis
- Classification Type: Binary Classification
- Classes: Benign (0) and Malignant (1)

The dataset was divided into training and testing sets. The test dataset contains 114 records.

## C. GitHub Repository Link

GitHub Repository: To be added after repository creation.

## D. Models Used

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation Metrics

The following six metrics were used:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


### Model Comparison and Evaluation Results

#### 1. Logistic Regression
- Accuracy: 0.9649
- AUC Score: 0.9960
- Precision: 0.9750
- Recall: 0.9286
- F1 Score: 0.9512
- MCC Score: 0.9245

**Observation:** Logistic Regression achieved excellent performance with high accuracy and AUC. It provides a strong balance between precision and recall.

#### 2. Decision Tree Classifier
- Accuracy: 0.9298
- AUC Score: 0.9246
- Precision: 0.9048
- Recall: 0.9048
- F1 Score: 0.9048
- MCC Score: 0.8492

**Observation:** Decision Tree achieved reasonable classification performance. However, its overall performance was lower than the other leading models.

#### 3. K-Nearest Neighbors (KNN)
- Accuracy: 0.9561
- AUC Score: 0.9823
- Precision: 0.9744
- Recall: 0.9048
- F1 Score: 0.9383
- MCC Score: 0.9058

**Observation:** KNN achieved high accuracy and precision. It performed well in distinguishing between benign and malignant cases.

#### 4. Gaussian Naive Bayes
- Accuracy: 0.9211
- AUC Score: 0.9891
- Precision: 0.9231
- Recall: 0.8571
- F1 Score: 0.8889
- MCC Score: 0.8292

**Observation:** Naive Bayes achieved a high AUC score but comparatively lower recall and F1 score. It performed reasonably well overall.

#### 5. Support Vector Machine (SVM)
- Accuracy: 0.9737
- AUC Score: 0.9947
- Precision: 1.0000
- Recall: 0.9286
- F1 Score: 0.9630
- MCC Score: 0.9442

**Observation:** SVM achieved the highest accuracy, precision, F1 score and MCC score among all five models. It demonstrated the strongest overall classification performance.

### Overall Winner

**Support Vector Machine (SVM)** is the overall winner based on its highest F1 score of 0.9630. It also achieved the highest accuracy of 0.9737 and MCC score of 0.9442.


## E. Streamlit Application Features

The application includes:

1. CSV test dataset upload.
2. Machine Learning model selection dropdown.
3. Display of six evaluation metrics.
4. Confusion matrix visualization.
5. Classification report.

The application loads previously trained models and evaluates uploaded test data without retraining.

## F. Project Structure

```text
ML_Classification_Assignment/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── train_data.csv
├── breast_cancer_data.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── results/
│
├── 01_dataset.py
├── 02_preprocessing.py
├── 03_train_models.py
├── 04_evaluation.py
└── 05_visualization.py

```

## G. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
