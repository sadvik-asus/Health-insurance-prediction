import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib

def main():
    print("Loading data...")
    df = pd.read_csv('train.csv')
    
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
        
    print("Preprocessing data...")
    df_encoded = df.copy()
    
    # Label Encoding
    le_gender = LabelEncoder()
    df_encoded['Gender'] = le_gender.fit_transform(df_encoded['Gender'])
    
    le_damage = LabelEncoder()
    df_encoded['Vehicle_Damage'] = le_damage.fit_transform(df_encoded['Vehicle_Damage'])
    
    # Map Vehicle_Age
    age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    df_encoded['Vehicle_Age'] = df_encoded['Vehicle_Age'].map(age_map)
    
    # Features and Target
    X = df_encoded.drop('Response', axis=1)
    y = df_encoded['Response']
    
    # Train-test split (we could train on all data, but let's stick to the script's approach for validation if needed, or just fit on all)
    print("Scaling continuous features...")
    scaler = StandardScaler()
    cols_to_scale = ['Age', 'Annual_Premium', 'Vintage']
    
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    
    print("Training XGBoost model...")
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight, 
        use_label_encoder=False, 
        eval_metric='logloss', 
        learning_rate=0.1, 
        n_estimators=200, 
        max_depth=6, 
        random_state=42, 
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    print("Saving model and preprocessors...")
    joblib.dump(model, 'xgboost_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(le_gender, 'le_gender.joblib')
    joblib.dump(le_damage, 'le_damage.joblib')
    
    print("Done! Artifacts saved.")

if __name__ == '__main__':
    main()
