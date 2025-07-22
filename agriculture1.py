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
    df = pd.read_csv('crop_insurance.csv')
except FileNotFoundError:
    print("FATAL ERROR: 'crop_insurance.csv' not found. Please ensure it is in the correct directory.")
    exit()

# 2. Preprocess the Data
clean_column_map = {
    'Crop': 'Crop',
    'State': 'State',
    'Cost of Cultivation (`/Hectare) A2+FL': 'Cost_Cultivation_A2_FL',
    'Cost of Cultivation (`/Hectare) C2': 'Cost_Cultivation_C2',
    'Cost of Production (`/Quintal) C2': 'Cost_Production_C2',
    'Yield (Quintal/ Hectare) ': 'Yield_Quintal_Hectare'
}
df.rename(columns=clean_column_map, inplace=True)
df.dropna(inplace=True)

# Define target and features
target = 'Yield_Quintal_Hectare'
y = df[target]
features = df.drop(columns=[target])
X_encoded = pd.get_dummies(features, columns=['Crop', 'State'], drop_first=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 4. Initialize and Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=7,         # Restricting depth to avoid overfitting
    random_state=42,
    n_jobs=-1
)

print("Training the Random Forest model...")
model.fit(X_train, y_train)
print("Training complete.\n")

# 5. Evaluate the Model
print("--- Model Evaluation ---")
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"R-squared (R²) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f} Quintal/Hectare")
print("------------------------\n")

# 6. Predict on New Data
print("--- Example Prediction ---")
new_crop_data = {
    'Cost_Cultivation_A2_FL': 29047.10,
    'Cost_Cultivation_C2': 50828.83,
    'Cost_Production_C2': 2003.76,
    'Crop_COTTON': True,
    'State_Punjab': True
}
new_df = pd.DataFrame([new_crop_data])
final_df = new_df.reindex(columns=X_train.columns, fill_value=False)
predicted_yield = model.predict(final_df)

print(f"Data for new crop: {new_crop_data}")
print(f"Predicted Yield: {predicted_yield[0]:,.2f} Quintal/Hectare")
print("------------------------")
