import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import random
import os

# --- MEDICAL MODEL ---
def train_and_save_medical_model():
    print("Training medical model...")
    try:
        df = pd.read_csv('insurance.csv')
        y = df['charges']
        features = df.drop('charges', axis=1)
        X_encoded = pd.get_dummies(features, columns=['sex', 'smoker', 'region'], drop_first=True)

        # Save the feature columns
        joblib.dump(X_encoded.columns, 'medical_model_columns.pkl')

        # Train the Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_encoded, y)

        # Save the model
        joblib.dump(model, 'medical_model.pkl')
        print("Medical model saved successfully.")
    except FileNotFoundError:
        print("ERROR: insurance.csv not found. Skipping medical model training.")

def predict_charge(data):
    model = joblib.load('medical_model.pkl')
    model_columns = joblib.load('medical_model_columns.pkl')
    input_df = pd.DataFrame([data])
    final_df = input_df.reindex(columns=model_columns, fill_value=0)
    return model.predict(final_df)[0]

# --- AGRICULTURE MODEL ---
def train_and_save_agriculture_model():
    print("Training agriculture model...")
    try:
        df = pd.read_csv('crop_insurance.csv')

        # Clean and rename columns
        df.rename(columns={
            'Cost of Cultivation (`/Hectare) C2': 'Cost_Cultivation_C2',
            'Cost of Production (`/Quintal) C2': 'Cost_Production_C2',
            'Yield (Quintal/ Hectare) ': 'Yield'
        }, inplace=True)
        df.dropna(inplace=True)

        y = df['Yield']
        features = df.drop('Yield', axis=1)
        X_encoded = pd.get_dummies(features, columns=['Crop', 'State'])

        # Save feature columns
        joblib.dump(X_encoded.columns, 'agriculture_model_columns.pkl')

        # Train the Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_encoded, y)

        # Save model
        joblib.dump(model, 'agriculture_model.pkl')
        print("Agriculture model saved successfully.")
    except FileNotFoundError:
        print("ERROR: crop_insurance.csv not found. Skipping agriculture model training.")

def predict_yield(data):
    model = joblib.load('agriculture_model.pkl')
    model_columns = joblib.load('agriculture_model_columns.pkl')
    input_df = pd.DataFrame([data])
    final_df = input_df.reindex(columns=model_columns, fill_value=0)
    return model.predict(final_df)[0]

# --- PROPERTY MODEL (Placeholder) ---
def predict_property_risk(data):
    print("Generating placeholder property prediction...")
    return random.choice(['Low Risk', 'Medium Risk', 'High Risk'])

# --- Main execution block ---
if __name__ == '__main__':
    print("--- Starting Model Training ---")
    train_and_save_medical_model()
    train_and_save_agriculture_model()
    print("--- Model Training Complete ---")
