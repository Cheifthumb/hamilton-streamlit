# train_xgboost_tuned.py

import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_excel('featured_data/02-25_FEATURED.xlsx')

# Pre-race features
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
y = data['Place'].apply(lambda x: 1 if x == 1 else 0)

# Encode categorical variables
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.fillna(-999)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adjust imbalance
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# Tuned XGBoost model
model = xgb.XGBClassifier(
    n_estimators=1000,        # more trees
    learning_rate=0.02,       # slower learning
    max_depth=5,              # slightly shallower trees
    subsample=0.8,            # random sampling
    colsample_bytree=0.7,     # random feature selection
    reg_alpha=1,              # L1 regularization (sparsity)
    reg_lambda=1,             # L2 regularization (ridge)
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ AUC Score:", roc_auc_score(y_test, y_prob))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'models/xgboost_tuned_model_02-25.pkl')
print("\n✅ Model saved to 'models/xgboost_tuned_model.pkl'")

