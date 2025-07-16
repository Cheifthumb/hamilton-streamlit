# save_predictions.py

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load your full featured dataset
data = pd.read_excel('featured_data/23-23_featured.xlsx')

# Info columns to save (include real odds and Place)
info_columns = [
    'Date of Race', 'Time', 'Track', 'Horse', 'Distance',
    'Place', 'Industry SP', 'Betfair SP'
]

# Pre-race features used for prediction
pre_race_features = [
    'Going', 'Distance', 'Class', 'Stall', 'Official Rating', 'Age',
    'SP Fav', 'Industry SP', 'Forecasted Odds',
    'Runs last 18 months', 'Wins Last 5 races',
    'Avg % SP Drop Last 5 races', 'Avg % SP Drop last 18 mths',
    'RBD Rating', 'RBD Rank', 'Total Prev Races',
    'Days Since Last time out', 'Course Wins', 'Distance Wins', 'Class Wins', 'Going Wins',
    'Up in Trip'
]

# Filter features
existing_features = [feature for feature in pre_race_features if feature in data.columns]
X = data[existing_features]

# Encode text features
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.fillna(-999)

import joblib

# Load trained Kempton model
model = joblib.load('models/xgboost_tuned_model_02-22.pkl')  # ✅ Update the correct filename and path


# Predict probabilities
win_probabilities = model.predict_proba(X)[:, 1]

# Build output
output = data[info_columns].copy()
output['Predicted_Win_Probability'] = win_probabilities

# Save to CSV
output.to_csv('Betting_Simulation/predicted_win_probabilities.23-23.csv', index=False)
print("✅ Predictions saved with real odds to 'Betting_Simulation.csv'")
