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
* **Data Preprocessing & Encoding:** 
  * **Missing Values & Duplicates:** Verified data integrity (no missing values were found) and dropped the irrelevant `id` column.
  * **Categorical Encoding:** Applied `LabelEncoder` for binary variables (`Gender`, `Vehicle_Damage`) and mapped `Vehicle_Age` manually to preserve its ordinal nature (`< 1 Year`: 0, `1-2 Year`: 1, `> 2 Years`: 2).
  * **Feature Scaling:** Applied targeted Feature Scaling (`StandardScaler`) exclusively to continuous numerical columns (`Age`, `Annual_Premium`, `Vintage`) to prevent distortion of encoded categorical variables.
  * **Data Subsetting:** Utilized 100% of the dataset for training to maximize information retention instead of downsampling.
* **Machine Learning Models & Hyperparameter Tuning:** 
  * **Logistic Regression:** Used as a baseline model, trained with `class_weight='balanced'`.
  * **Random Forest Classifier:** An ensemble tree model, tuned with `n_estimators=200`, `max_depth=12`, and `class_weight='balanced'` to prevent overfitting and handle class distribution.
  * **XGBoost Classifier:** A powerful gradient boosting model, tuned with `learning_rate=0.1`, `n_estimators=200`, `max_depth=6`. The exact positive class ratio was calculated and passed to `scale_pos_weight` to perfectly handle the imbalanced labels.

## Project Structure
* `health_insurance_project.ipynb`: A Jupyter Notebook containing step-by-step code, documentation, and visualizations for the entire ML pipeline.
* `health_insurance_project.py`: A streamlined Python script of the end-to-end pipeline.

## Evaluation & Results
Due to the highly imbalanced nature of the dataset (~12% positive class), traditional Accuracy is heavily misleading and not a reliable metric. Instead, models were evaluated and compared primarily based on **ROC-AUC**, **F1-Score**, Precision, and Recall.

**Results Summary:**
* **XGBoost** performed the best out of the three models, effectively capturing complex non-linear relationships and achieving the highest ROC-AUC score.
* The explicit handling of class imbalances drastically improved the F1-Score across all models compared to default unweighted implementations.

## How to Run
1. Clone the repository.
2. Download the [Kaggle Dataset](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) and place the `train.csv` file in the root directory.
3. Run the Jupyter Notebook `health_insurance_project.ipynb` to view the analysis and train the models.
