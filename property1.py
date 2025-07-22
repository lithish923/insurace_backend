import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress all warnings for a clean output
warnings.filterwarnings('ignore')

# 1. Load the Built-in Dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# 2. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,     # Number of trees
    max_depth=None,       # Full depth (can be tuned)
    random_state=42,
    n_jobs=-1             # Use all processors
)

# Train the model
print("Training a high-accuracy Random Forest Classifier...")
model.fit(X_train, y_train)
print("Training complete.\n")

# 4. Evaluate the Model's Performance
print("--- Model Evaluation ---")
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=cancer.target_names))
print("------------------------\n")

# 5. Predict on New Data (Example)
print("--- Example Prediction ---")
new_sample = X_test.iloc[[0]]
predicted_class_code = model.predict(new_sample)
predicted_class_name = cancer.target_names[predicted_class_code[0]]

print("Example sample data (first 5 features):")
print(new_sample.iloc[:, :5])
print(f"\nPredicted Diagnosis: {predicted_class_name.capitalize()}")
print("------------------------")
