"""
train_and_export.py

Run this ONCE in the same environment as your notebook (Colab / Jupyter / local),
in the same folder as P685_alzheimers_disease_data.csv.

It re-trains the final tuned Random Forest model on the 7 selected features
and saves it as model/model.pkl, ready to be used by the Streamlit app (streamlit_app.py).
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---- 1. Load data ----
df = pd.read_csv('P685_alzheimers_disease_data (1).csv')

# ---- 2. Final 7 features chosen after voting (Chi-Square + T-Test + MI + RF) ----
FEATURES = [
    "MemoryComplaints",
    "BehavioralProblems",
    "SleepQuality",
    "CholesterolHDL",
    "MMSE",
    "FunctionalAssessment",
    "ADL",
]
TARGET = "Diagnosis"

X = df[FEATURES]
y = df[TARGET]

# ---- 3. Same stratified 80/20 split used in the notebook ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ---- 4. Train Random Forest with the best parameters found via GridSearchCV ----
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42,
)
model.fit(X_train, y_train)

# ---- 5. Quick sanity check on the held-out test set ----
acc = model.score(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# ---- 6. Save the trained model + feature importances for the app ----
import os
import joblib

# Get the absolute path of the directory where this script lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths for the model and features
model_dir = os.path.join(current_dir, "model")
model_path = os.path.join(model_dir, "model.pkl")
features_path = os.path.join(model_dir, "features.pkl")

# Ensure the 'model' folder exists right next to your script
os.makedirs(model_dir, exist_ok=True)

# Save the files
joblib.dump(model, model_path)
joblib.dump(FEATURES, features_path)

print(f"Successfully saved to: {model_path}")
