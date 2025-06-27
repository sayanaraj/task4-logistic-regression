# task4-logistic-regression

## Objective

The goal of this task is to build a binary classification model using Logistic Regression. The model is trained to predict a binary outcome (e.g., malignant vs benign tumors) using real-world data. The task also emphasizes understanding key classification metrics such as precision, recall, confusion matrix, and ROC-AUC, as well as interpreting the sigmoid function and decision threshold.

## Dataset

- **Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: [Kaggle - UCI Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **File Used**: `data.csv`

The dataset includes features computed from digitized images of breast mass biopsies. It contains:
- 30 numeric features (e.g., radius, texture, smoothness)
- A target column `diagnosis` with labels `M` (malignant) or `B` (benign)

## Tools and Libraries

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
  
## Steps Performed

### 1. Data Preprocessing
- Loaded the dataset using `pandas`.
- Dropped irrelevant columns like `id` and unnamed columns.
- Converted the target column `diagnosis` to binary format: `M = 1`, `B = 0`.
- Checked for and confirmed there were no missing values.
- Standardized the features using `StandardScaler` for better model convergence.

### 2. Train-Test Split
- Split the dataset into training and testing sets using `train_test_split` (80% train, 20% test).

### 3. Model Training
- Trained a `LogisticRegression` model from `scikit-learn` on the training set.

### 4. Model Evaluation
- Evaluated model performance using:
  - **Confusion Matrix**
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **ROC-AUC Score**

### 5. Threshold Tuning
- Default threshold of 0.5 used for classification.
- Custom threshold tuning demonstrated by adjusting sigmoid probabilities manually.

### 6. Visualization
- Confusion matrix heatmap using `seaborn`.
- ROC Curve using `sklearn.metrics.roc_curve`.
- Sigmoid function curve plotted for interpretation.
- 
## Key Concepts Explained

### Logistic Regression vs Linear Regression
- Linear Regression predicts continuous values; Logistic Regression predicts binary classes using a probability and threshold.

### Sigmoid Function
- Maps real-valued inputs to a range between 0 and 1.
- Used in logistic regression to convert linear outputs into probabilities.

### Confusion Matrix
- Shows TP, FP, TN, FN and is used to calculate other metrics.

### Precision vs Recall
- **Precision**: How many predicted positives are actual positives.
- **Recall**: How many actual positives were predicted correctly.

### ROC-AUC Curve
- Plots True Positive Rate vs False Positive Rate.
- AUC (Area Under Curve) closer to 1 indicates better performance.

