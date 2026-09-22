from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and preprocessors
try:
    model = joblib.load('xgboost_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le_gender = joblib.load('le_gender.joblib')
    le_damage = joblib.load('le_damage.joblib')
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features
        gender = data.get('Gender', 'Male')
        age = float(data.get('Age', 30))
        driving_license = int(data.get('Driving_License', 1))
        region_code = float(data.get('Region_Code', 28.0))
        previously_insured = int(data.get('Previously_Insured', 0))
        vehicle_age = data.get('Vehicle_Age', '1-2 Year')
        vehicle_damage = data.get('Vehicle_Damage', 'Yes')
        annual_premium = float(data.get('Annual_Premium', 30000.0))
        policy_sales_channel = float(data.get('Policy_Sales_Channel', 124.0))
        vintage = float(data.get('Vintage', 150.0))
        
        # Preprocess exactly as in training
        # Gender
        gender_encoded = le_gender.transform([gender])[0]
        # Vehicle Damage
        damage_encoded = le_damage.transform([vehicle_damage])[0]
        # Vehicle Age Map
        age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
        vehicle_age_encoded = age_map.get(vehicle_age, 1)
        
        # Create DataFrame
        df_input = pd.DataFrame([[
            gender_encoded, age, driving_license, region_code, 
            previously_insured, vehicle_age_encoded, damage_encoded, 
            annual_premium, policy_sales_channel, vintage
        ]], columns=[
            'Gender', 'Age', 'Driving_License', 'Region_Code', 
            'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 
            'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'
        ])
        
        # Scale continuous features
        cols_to_scale = ['Age', 'Annual_Premium', 'Vintage']
        df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])
        
        # Predict
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
