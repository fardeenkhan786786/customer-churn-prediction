# Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a customer is likely to churn.

## Project Overview

This project uses customer information such as tenure, contract type, internet service, monthly charges, payment method, and other features to predict customer churn.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## Machine Learning Model

The project uses Logistic Regression for binary classification.

## Workflow

1. Data loading
2. Data cleaning
3. Exploratory Data Analysis
4. Feature preprocessing
5. Train-test split
6. Model training
7. Model evaluation
8. Model saving
9. Streamlit deployment

## Model Performance

ROC-AUC Score: 0.8357

## Application

A Streamlit web application allows users to enter customer information and receive:

- Churn prediction
- Churn probability
- High/Low churn risk

## Project Structure

customer-churn-prediction/
│
├── data/
│   └── customer_churn.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   ├── churn_model.pkl
│   └── preprocessor.pkl
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt