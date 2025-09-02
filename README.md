# Customer Churn Prediction Using ANN

## Project Overview
This project aims to predict customer churn using an Artificial Neural Network (ANN) on a bank customer dataset. The dataset contains customer information and whether they exited the bank or not. The model uses data preprocessing, oversampling techniques, and a focal loss function to handle class imbalance and improve predictive performance.

---

## Dataset
The dataset used is `Churn_Modelling.csv`, which contains 10,000 records with features such as:

- Customer demographics (Age, Gender, Geography)
- Account details (CreditScore, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
- Target variable: `Exited` (1 if customer churned, 0 otherwise)

---

## Project Structure
- Data Loading and Exploration
- Data Preprocessing
  - Handling categorical variables (encoding)
  - Handling outliers with IQR capping
- Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
- Feature Scaling using StandardScaler
- ANN model building with TensorFlow/Keras
  - Using Focal Loss to address class imbalance
- Model training and validation
- Evaluation with accuracy and classification report

---

## How to Run

1. **Install required libraries**:

```bash
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn tensorflow
