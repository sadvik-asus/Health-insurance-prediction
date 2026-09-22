# Health Insurance Cross Sell Prediction

## Overview
This repository contains an end-to-end data science pipeline that predicts whether health insurance customers will be interested in purchasing vehicle insurance. The goal is to build an accurate predictive model that helps the insurance company target the right customers, optimizing their cross-selling strategy and maximizing revenue.

## Dataset Details
The dataset consists of various customer demographics, vehicle information, and policy details.
* **Numerical Columns:** `Age`, `Annual_Premium`, `Vintage`
* **Categorical Columns:** `Gender`, `Driving_License`, `Region_Code`, `Previously_Insured`, `Vehicle_Age`, `Vehicle_Damage`, `Policy_Sales_Channel`
* **Target Variable:** `Response` (1: Customer is interested, 0: Customer is not interested)

## Key Features
* **Exploratory Data Analysis (EDA):** Visualized feature distributions and analyzed the relationship between categorical/numerical features and the target variable.
* **Data Preprocessing:** 
  * Addressed severe class imbalance by utilizing the full dataset.
  * Preserved categorical variables while applying targeted Feature Scaling (StandardScaler) exclusively to continuous numerical columns (Age, Annual_Premium, Vintage).
* **Machine Learning Models:** 
  * Trained and evaluated **Logistic Regression**, **Random Forest**, and **XGBoost**.
  * Handled imbalanced classes dynamically using `class_weight='balanced'` and `scale_pos_weight`.
  * Hyperparameter tuning applied to ensemble models to extract maximum predictive power.

## Project Structure
* `health_insurance_project.ipynb`: A Jupyter Notebook containing step-by-step code, documentation, and visualizations for the entire ML pipeline.
* `health_insurance_project.py`: A streamlined Python script of the end-to-end pipeline.

## Evaluation
Due to the highly imbalanced nature of the dataset, traditional Accuracy is not a reliable metric. The models in this project are evaluated and compared primarily based on **ROC-AUC** and **F1-Score**.

## How to Run
1. Clone the repository.
2. Download the [Kaggle Dataset](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) and place the `train.csv` file in the root directory.
3. Run the Jupyter Notebook `health_insurance_project.ipynb` to view the analysis and train the models.
