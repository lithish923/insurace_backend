import pandas as pd
import joblib
import random
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os

app = Flask(__name__)
CORS(app)

# === MEDICAL MODEL ===
def train_medical_model():
    try:
        df = pd.read_csv('insurance.csv')
        y = df['charges']
        X = pd.get_dummies(df.drop('charges', axis=1), columns=['sex', 'smoker', 'region'], drop_first=True)

        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X, y)

        joblib.dump(model, 'medical_model.pkl')
        joblib.dump(X.columns, 'medical_model_columns.pkl')
        print("✅ Medical model trained and saved.")
    except FileNotFoundError:
        print("❌ ERROR: 'insurance.csv' not found. Skipping medical model training.")

def predict_charge(data):
    model = joblib.load('medical_model.pkl')
    model_columns = joblib.load('medical_model_columns.pkl')
    input_df = pd.DataFrame([data])
    final_df = input_df.reindex(columns=model_columns, fill_value=0)
    return model.predict(final_df)[0]


# === AGRICULTURE MODEL ===
def train_agriculture_model():
    try:
        df = pd.read_csv('crop_insurance.csv')
        df.rename(columns={
            'Cost of Cultivation (`/Hectare) C2': 'Cost_Cultivation_C2',
            'Cost of Production (`/Quintal) C2': 'Cost_Production_C2',
            'Yield (Quintal/ Hectare) ': 'Yield',
            'Cost of Cultivation (`/Hectare) A2+FL': 'Cost_Cultivation_A2_FL'
        }, inplace=True)
        df.dropna(inplace=True)
        y = df['Yield']
        X = pd.get_dummies(df.drop('Yield', axis=1), columns=['Crop', 'State'])

        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X, y)

        joblib.dump(model, 'agriculture_model.pkl')
        joblib.dump(X.columns, 'agriculture_model_columns.pkl')
        print("✅ Agriculture model trained and saved.")
    except FileNotFoundError:
        print("❌ ERROR: 'crop_insurance.csv' not found. Skipping agriculture model training.")

def predict_yield(data):
    model = joblib.load('agriculture_model.pkl')
    model_columns = joblib.load('agriculture_model_columns.pkl')
    input_df = pd.DataFrame([data])
    final_df = input_df.reindex(columns=model_columns, fill_value=0)
    return model.predict(final_df)[0]


# === PROPERTY MODEL (Placeholder) ===
def predict_property_risk(data):
    return random.choice(['Low Risk', 'Medium Risk', 'High Risk'])


# === API ENDPOINTS ===

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'message': 'Backend is running'}), 200

@app.route('/predict/medical', methods=['POST'])
def medical_prediction():
    try:
        data = request.get_json()
        input_data = {
            'age': int(data.get('age') or 0),
            'bmi': float(data.get('bmi') or 0.0),
            'children': int(data.get('children') or 0),
            'sex_male': data.get('sex') == 'male',
            'smoker_yes': data.get('smoker') == 'yes',
            'region_northwest': data.get('region') == 'northwest',
            'region_southeast': data.get('region') == 'southeast',
            'region_southwest': data.get('region') == 'southwest',
        }
        predicted_charge = predict_charge(input_data)
        if predicted_charge < 8000: risk = 'Low Risk'
        elif predicted_charge < 20000: risk = 'Medium Risk'
        else: risk = 'High Risk'
        return jsonify({'risk_level': risk, 'predicted_value': round(predicted_charge, 2)})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict/agriculture', methods=['POST'])
def agriculture_prediction():
    try:
        data = request.get_json()
        state_feature = f"State_{data.get('state_name')}"
        crop_feature = f"Crop_{data.get('crop_name')}"
        input_data = {
            'Cost_Cultivation_A2_FL': float(data.get('cost_cultivation_a2fl') or 0.0),
            'Cost_Cultivation_C2': float(data.get('cost_cultivation_c2') or 0.0),
            'Cost_Production_C2': float(data.get('cost_production_c2') or 0.0),
            state_feature: True,
            crop_feature: True
        }
        predicted_yield = predict_yield(input_data)
        if predicted_yield > 25: risk = 'Low Risk'
        elif predicted_yield > 10: risk = 'Medium Risk'
        else: risk = 'High Risk'
        return jsonify({'risk_level': risk, 'predicted_value': round(predicted_yield, 2)})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict/property', methods=['POST'])
def property_prediction():
    data = request.get_json()
    risk = predict_property_risk(data)
    return jsonify({'risk_level': risk})


# === MAIN ===
if __name__ == '__main__':
    if not os.path.exists('medical_model.pkl') or not os.path.exists('agriculture_model.pkl'):
        print("🔄 Models not found. Training now...")
        train_medical_model()
        train_agriculture_model()
    else:
        print("✅ Models found. Skipping training.")

    app.run(host='0.0.0.0', port=5000, debug=True)
