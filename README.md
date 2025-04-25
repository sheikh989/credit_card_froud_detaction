# Credit Card Fraud Detection using Neural Networks

This repository presents a complete machine learning pipeline to detect fraudulent credit card transactions using a neural network. It includes comprehensive steps such as data preprocessing, feature encoding, model building, evaluation, and saving artifacts for future inference and deployment.

# Project Overview

Credit card fraud is a major concern for financial institutions around the world. The goal of this project is to build a deep learning model that can accurately identify fraudulent transactions based on available transaction and customer data. The dataset contains information such as the amount spent, location, merchant category, and personal attributes of the cardholder.

This project demonstrates how to build a robust fraud detection system using neural networks in TensorFlow/Keras, with careful attention to preprocessing, class imbalance, and evaluation metrics like precision and recall.

# Dataset Description

The dataset used is sourced from Kaggle: Fraud Detection Dataset
It consists of two files:

-fraudTrain.csv: Training dataset
-fraudTest.csv: Testing dataset

Each transaction contains features like:

-Amount

-Category

-Merchant

-Card number

-City and state

-Job

-Gender

-Date and time

-Binary target variable is_fraud

# Required packages include:

-numpy

-pandas

-matplotlib

-seaborn

-scikit-learn

-pickle

-kagglehub (for loading dataset)

# Data Preprocessing

-Unnecessary columns such as names, transaction ID, and date-time are dropped.

-Outliers in numerical features like amt and city_pop are capped using the IQR method.

-Categorical features are encoded using LabelEncoder.

-Gender is converted to binary (Male: 1, Female: 0).

-All features are normalized using MinMaxScaler.

# Modeling
Three machine learning algorithms are used:

-Logistic Regression

-Decision Tree Classifier

-Random Forest Classifier

-Train-test split is applied to evaluate model generalization.

# Evaluation Metrics

The following metrics are used for model evaluation:

-Accuracy

-Precision

-Recall

-F1 Score

-Confusion Matrix

-Due to class imbalance, more focus is given to precision and recall rather than just accuracy.

# Results
Random Forest performs better than Logistic Regression and Decision Tree in terms of precision and recall. Proper preprocessing, feature selection, and model tuning contribute significantly to improved results.

# Future Work

Using SMOTE or undersampling to handle class imbalance

# License

This project is open-source and available under the MIT License.
