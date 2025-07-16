import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

#python -m streamlit run betting_web_app.py


# Streamlit config
st.set_page_config(page_title="Horse Racing Bets", layout="wide")
st.title("🏇 Horse Racing Prediction and Bet Selector")

uploaded_file = st.file_uploader("📤 Upload your Excel race file", type=["xlsx"])
output_name = ""

if uploaded_file:
    file_name = uploaded_file.name
    output_name = file_name.replace(".xlsx", "")
    input_path = f"Daily_Bets/{file_name}"

    os.makedirs("Daily_Bets", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {file_name}")

    # === CLEANING & FEATURE ENGINEERING ===
    df = pd.read_excel(input_path)

    df['Date of Race'] = pd.to_datetime(df['Date of Race'], errors='coerce')
    df = df[df['Date of Race'].notnull()]
    df['Industry SP'] = pd.to_numeric(df['Industry SP'], errors='coerce')
    df = df[df['Industry SP'].notnull()]

    numeric_cols = [
        'Forecasted Odds', 'Industry SP', 'SP Win Return',
        'Betfair Lay Return', 'Wins Last 5 races', 'Runs last 18 months',
        'Course Wins', 'Distance Wins', 'Total Prev Races'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Days Since Last time out' not in df.columns and 'Date of Race' in df.columns and 'Horse' in df.columns:
        df = df.sort_values(['Horse', 'Date of Race'])
        df['Days Since Last time out'] = df.groupby('Horse')['Date of Race'].diff().dt.days

    df['Log Industry SP'] = df['Industry SP'].apply(lambda x: pd.NA if pd.isna(x) or x <= 0 else np.log(x))
    if 'SP Fav' in df.columns:
        df['Is Favourite'] = df['SP Fav'].apply(lambda x: 1 if str(x).strip().lower() == 'fav' else 0)
    if 'Industry SP' in df.columns and 'Track' in df.columns:
        df['SP Rank'] = df.groupby(['Date of Race', 'Track'])['Industry SP'].rank(method='min')
    if 'Up in Trip' in df.columns:
        df['Up in Trip'] = df['Up in Trip'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    if 'Wins Last 5 races' in df.columns and 'Total Prev Races' in df.columns:
        df['Win Rate Last 5'] = df['Wins Last 5 races'] / df['Total Prev Races'].replace(0, pd.NA)
    if 'Wins Last 5 races' in df.columns and 'Runs last 18 months' in df.columns:
        df['Adjusted Win Rate'] = df['Wins Last 5 races'] / df['Runs last 18 months'].replace(0, pd.NA)
    if 'Course Wins' in df.columns and 'Total Prev Races' in df.columns:
        df['Course Win Ratio'] = df['Course Wins'] / df['Total Prev Races'].replace(0, pd.NA)
    if 'Distance Wins' in df.columns and 'Total Prev Races' in df.columns:
        df['Distance Win Ratio'] = df['Distance Wins'] / df['Total Prev Races'].replace(0, pd.NA)
    if 'Class' in df.columns:
        df['Class'] = pd.to_numeric(df['Class'], errors='coerce')
    if 'Forecasted Odds' in df.columns and 'Industry SP' in df.columns:
        df['Value Indicator'] = df['Forecasted Odds'] - df['Industry SP']
    if 'Betfair Lay Return' in df.columns and 'SP Win Return' in df.columns:
        df['Lay Pressure %'] = (df['Betfair Lay Return'] - df['SP Win Return']) / df['SP Win Return'].replace(0, pd.NA)

    # === PREDICTIONS ===
    info_columns = ['Date of Race', 'Time', 'Track', 'Horse', 'Distance', 'Place', 'Industry SP', 'Betfair SP']
    pre_race_features = [
        'Going', 'Distance', 'Class', 'Stall', 'Official Rating', 'Age',
        'SP Fav', 'Industry SP', 'Forecasted Odds',
        'Runs last 18 months', 'Wins Last 5 races',
        'Avg % SP Drop Last 5 races', 'Avg % SP Drop last 18 mths',
        'RBD Rating', 'RBD Rank', 'Total Prev Races',
        'Days Since Last time out', 'Course Wins', 'Distance Wins', 'Class Wins', 'Going Wins',
        'Up in Trip'
    ]
    existing_features = [feature for feature in pre_race_features if feature in df.columns]
    X = df[existing_features]

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(-999)

    model = joblib.load("models/xgboost_tuned_model_02-25.pkl")
    win_probabilities = model.predict_proba(X)[:, 1]

    predictions = df[info_columns].copy()
    predictions['Predicted_Win_Probability'] = win_probabilities

    # === BET FILTERING ===
    predictions['Race_ID'] = predictions['Date of Race'].astype(str) + "_" + predictions['Time'].astype(str)
    predictions['Odds_To_Use'] = predictions['Industry SP']
    predictions['Predicted_Win_Probability'] = predictions.groupby('Race_ID')['Predicted_Win_Probability'].transform(lambda x: x / x.sum())
    predictions['Predicted_Rank'] = predictions.groupby('Race_ID')['Predicted_Win_Probability'].rank(method='first', ascending=False)

    # Get full field size BEFORE filtering
    field_sizes = predictions.groupby('Race_ID')['Horse'].count().rename('Field_Size')
    predictions = predictions.merge(field_sizes, left_on='Race_ID', right_index=True)

    # Strategy settings
    rank_filter_enabled = True
    allowed_predicted_ranks = [1, 2]
    daily_bankroll = 10000
    bankroll_perc = 0.1
    stake_pool = daily_bankroll * bankroll_perc
    min_ev_threshold = 0.10
    min_kelly_fraction = 0.01
    max_odds_threshold = 100.0
    winrate_filter_type = 'none'
    fixed_winrate_threshold = 0.03
    min_runners = 5
    max_runners = 6

    predictions['Reject_Reason'] = ''
    if rank_filter_enabled:
        predictions.loc[~predictions['Predicted_Rank'].isin(allowed_predicted_ranks), 'Reject_Reason'] += 'rank_low|'

    # EV & Kelly
    p = predictions['Predicted_Win_Probability']
    b = predictions['Odds_To_Use'] - 1
    q = 1 - p
    predictions['Expected_Value'] = (p * b) - q
    predictions['Kelly_Fraction'] = (b * p - q) / b

    predictions.loc[predictions['Expected_Value'] <= min_ev_threshold, 'Reject_Reason'] += 'ev_low|'
    predictions.loc[predictions['Kelly_Fraction'] <= min_kelly_fraction, 'Reject_Reason'] += 'kelly_low|'
    predictions.loc[predictions['Odds_To_Use'] > max_odds_threshold, 'Reject_Reason'] += 'odds_high|'
    predictions.loc[(predictions['Field_Size'] < min_runners) | (predictions['Field_Size'] > max_runners), 'Reject_Reason'] += 'field_size|'

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
        (predictions['Predicted_Win_Probability'] > predictions['Winrate_Threshold']) &
        (predictions['Predicted_Rank'].isin(allowed_predicted_ranks))
    )

    # Stake assignment
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

    # Output: bets + rejected
    final_bets = predictions[predictions['Bet_Recommended']]
    st.subheader("✅ Final Bets")
    st.dataframe(final_bets[['Date of Race', 'Time', 'Horse', 'Industry SP', 'Predicted_Win_Probability',
                             'Expected_Value', 'Kelly_Fraction', 'Predicted_Rank', 'Field_Size', 'Recommended_Stake']])

    rejected_bets = predictions[~predictions['Bet_Recommended']]
    if not rejected_bets.empty:
        st.subheader("❌ Rejected Bets")
        st.dataframe(rejected_bets[['Date of Race', 'Time', 'Horse', 'Industry SP', 'Predicted_Win_Probability',
                                    'Expected_Value', 'Kelly_Fraction', 'Predicted_Rank', 'Field_Size', 'Reject_Reason']])
