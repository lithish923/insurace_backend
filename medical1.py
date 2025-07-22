import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

# Suppress all warnings for a clean output
warnings.filterwarnings('ignore')

# 1. Load the Dataset
try:
    df = pd.read_csv('insurance.csv')
except FileNotFoundError:
    print("FATAL ERROR: 'insurance.csv' not found. Please ensure it is in the correct directory.")
    exit()

# 2. Preprocess the Data
y = df['charges']
features = df.drop('charges', axis=1)

# 3. One-Hot Encode Categorical Features
X_encoded = pd.get_dummies(features, columns=['sex', 'smoker', 'region'], drop_first=True)

# 4. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. Initialize and Train the Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,  # Number of trees in the forest
    max_depth=10,      # Control tree depth (tune as needed)
    random_state=42,
    n_jobs=-1          # Use all CPU cores
)

print("Training the Random Forest model...")
model.fit(X_train, y_train)
print("Training complete.\n")

# 6. Evaluate the Model's Performance
print("--- Model Evaluation ---")
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"R-squared (R²) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): ₹{mae:,.2f}")
print("------------------------\n")

# 7. Predict on New Data (Example)
print("--- Example Prediction ---")
new_person = {
    'age': 35,
    'bmi': 25.0,
    'children': 1,
    'sex_male': True,
    'smoker_yes': False,
    'region_northwest': True,
    'region_southeast': False,
    'region_southwest': False
}

new_person_df = pd.DataFrame([new_person], columns=X_train.columns)
predicted_charge = model.predict(new_person_df)

print(f"Data for new person: {new_person}")
print(f"Predicted Insurance Charge: ₹{predicted_charge[0]:,.2f}")
print("------------------------")
