import argparse
import os
from Clean_and_feature import clean_and_engineer
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib

def run_pipeline(input_file, model_path='models/xgboost_tuned_model_02-25.pkl'):
    # Derive base name (e.g., '12.07.2025')
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # === STEP 1: Clean + Feature Engineer ===
    featured_file = clean_and_engineer(input_file, output_name=base_name)

    # === STEP 2: Predict Win Probabilities ===
    data = pd.read_excel(featured_file)

    info_columns = [
        'Date of Race', 'Time', 'Track', 'Horse', 'Distance',
        'Place', 'Industry SP', 'Betfair SP'
    ]

    pre_race_features = [
        'Going', 'Distance', 'Class', 'Stall', 'Official Rating', 'Age',
        'SP Fav', 'Industry SP', 'Forecasted Odds',
        'Runs last 18 months', 'Wins Last 5 races',
        'Avg % SP Drop Last 5 races', 'Avg % SP Drop last 18 mths',
        'RBD Rating', 'RBD Rank', 'Total Prev Races',
        'Days Since Last time out', 'Course Wins', 'Distance Wins', 'Class Wins', 'Going Wins',
        'Up in Trip'
    ]

    existing_features = [feature for feature in pre_race_features if feature in data.columns]
    X = data[existing_features].copy()

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.fillna(-999)

    model = joblib.load(model_path)
    win_probabilities = model.predict_proba(X)[:, 1]

    prediction_df = data[info_columns].copy()
    prediction_df['Predicted_Win_Probability'] = win_probabilities

    pred_path = f'Daily_Bets/predicted_win_probabilities.{base_name}.csv'
    prediction_df.to_csv(pred_path, index=False)
    print(f"✅ Predictions saved to {pred_path}")

    # === STEP 3: Apply Betting Logic ===

    # Settings
    daily_bankroll = 10000
    bankroll_perc = 0.1
    stake_pool = daily_bankroll * bankroll_perc
    min_ev_threshold = 0.00
    min_kelly_fraction = 0.00
    max_odds_threshold = 100.0
    winrate_filter_type = 'none'
    fixed_winrate_threshold = 0.03
    min_runners, max_runners = 1, 40
    rank_filter_enabled = True
    allowed_predicted_ranks = [1, 2]

    predictions = pd.read_csv(pred_path)
    predictions['Race_ID'] = predictions['Date of Race'].astype(str) + "_" + predictions['Time'].astype(str)
    predictions['Odds_To_Use'] = predictions['Industry SP']
    predictions['Predicted_Win_Probability'] = predictions.groupby('Race_ID')['Predicted_Win_Probability'].transform(lambda x: x / x.sum())
    predictions['Predicted_Rank'] = predictions.groupby('Race_ID')['Predicted_Win_Probability'].rank(method='first', ascending=False)

    if rank_filter_enabled:
        predictions = predictions[predictions['Predicted_Rank'].isin(allowed_predicted_ranks)]

    p = predictions['Predicted_Win_Probability']
    b = predictions['Odds_To_Use'] - 1
    q = 1 - p
    predictions['Expected_Value'] = (p * b) - q
    predictions['Kelly_Fraction'] = (b * p - q) / b

    field_sizes = predictions.groupby('Race_ID')['Horse'].count().rename('Field_Size')
    predictions = predictions.merge(field_sizes, left_on='Race_ID', right_index=True)
    predictions['Reject_Reason'] = ''

    predictions.loc[predictions['Expected_Value'] <= min_ev_threshold, 'Reject_Reason'] += 'ev_low|'
    predictions.loc[predictions['Kelly_Fraction'] <= min_kelly_fraction, 'Reject_Reason'] += 'kelly_low|'
    predictions.loc[predictions['Odds_To_Use'] > max_odds_threshold, 'Reject_Reason'] += 'odds_high|'
    predictions.loc[
        (predictions['Field_Size'] < min_runners) | (predictions['Field_Size'] > max_runners),
        'Reject_Reason'
    ] += 'field_size|'

    if winrate_filter_type == 'fixed':
        predictions['Winrate_Threshold'] = fixed_winrate_threshold
    elif winrate_filter_type == 'dynamic':
        predictions['Winrate_Threshold'] = 1 / predictions['Field_Size']
    else:
        predictions['Winrate_Threshold'] = 0

    predictions.loc[predictions['Predicted_Win_Probability'] <= predictions['Winrate_Threshold'], 'Reject_Reason'] += 'winrate_low|'

    predictions['Bet_Recommended'] = (
        (predictions['Expected_Value'] > min_ev_threshold) &
        (predictions['Kelly_Fraction'] > min_kelly_fraction) &
        (predictions['Odds_To_Use'] <= max_odds_threshold) &
        (predictions['Field_Size'] >= min_runners) &
        (predictions['Field_Size'] <= max_runners) &
        (predictions['Predicted_Win_Probability'] > predictions['Winrate_Threshold'])
    )

    predictions['Recommended_Stake'] = 0
    bets_to_place = []

    for race_id, race_df in predictions.groupby('Race_ID'):
        temp = race_df.copy()
        temp['Stake_Unscaled'] = temp['Kelly_Fraction'] * stake_pool
        total = temp.loc[temp['Bet_Recommended'], 'Stake_Unscaled'].sum()
        if total > stake_pool and total > 0:
            scale = stake_pool / total
            temp.loc[temp['Bet_Recommended'], 'Recommended_Stake'] = temp.loc[temp['Bet_Recommended'], 'Stake_Unscaled'] * scale
        else:
            temp.loc[temp['Bet_Recommended'], 'Recommended_Stake'] = temp.loc[temp['Bet_Recommended'], 'Stake_Unscaled']
        bets_to_place.append(temp)

    predictions = pd.concat(bets_to_place).reset_index(drop=True)

    predictions = predictions.sort_values(by=['Race_ID', 'Predicted_Win_Probability'], ascending=[True, False])
    predictions['Top_3_Pick_Rank'] = predictions.groupby('Race_ID').cumcount() + 1
    predictions['Top_3_Pick_Rank'] = predictions['Top_3_Pick_Rank'].where(predictions['Top_3_Pick_Rank'] <= 3)

    final_bets = predictions[predictions['Bet_Recommended'] == True].copy()
    final_bets = final_bets[['Date of Race', 'Time', 'Horse', 'Industry SP', 'Predicted_Win_Probability',
                             'Expected_Value', 'Kelly_Fraction', 'Top_3_Pick_Rank', 'Recommended_Stake']]

    final_path = f'Daily_Bets/Selection_{base_name}.csv'
    reject_path = f'Daily_Bets/Rejected_{base_name}.csv'

    final_bets.to_csv(final_path, index=False)
    predictions[predictions['Bet_Recommended'] == False].to_csv(reject_path, index=False)

    print(f"\n✅ Final bets saved to: {final_path}")
    print(f"❌ Rejected bets saved to: {reject_path}")
    print(f"📋 Preview:\n{final_bets}")

if __name__ == "__main__":
    print("📂 Welcome to the Horse Racing Full Pipeline Tool")
    input_file = input("📄 Please enter the full path or name of the race file (e.g., Daily_Bets/12.07.2025.xlsx): ").strip()
    
    if not input_file.lower().endswith(".xlsx"):
        print("❌ Invalid file type. Please provide a .xlsx file.")
    elif not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
    else:
        run_pipeline(input_file)

