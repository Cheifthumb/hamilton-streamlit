import pandas as pd

# ✅ Load and prepare data
data = pd.read_csv('Betting_Simulation/predicted_win_probabilities.24-25.csv')
data['Place'] = pd.to_numeric(data['Place'], errors='coerce')
data = data.sort_values(by=['Date of Race', 'Time']).reset_index(drop=True)
data['Race_ID'] = data['Date of Race'].astype(str) + "_" + data['Time'].astype(str)
data['Predicted_Rank'] = data.groupby('Race_ID')['Predicted_Win_Probability'].rank(method='first', ascending=False)

data['Odds_To_Use'] = data['Industry SP']
data['Predicted_Win_Probability'] = data.groupby('Race_ID')['Predicted_Win_Probability'].transform(lambda x: x / x.sum())
data['Expected_Value'] = (data['Predicted_Win_Probability'] * (data['Odds_To_Use'] - 1)) - (1 - data['Predicted_Win_Probability'])

# ✅ Settings
bankroll = 10000
bankroll_perc = 0.1
min_ev = 0.00
min_kelly = 0.00
max_odds = 100.0
winrate_filter_type = 'none'  # or 'fixed', 'dynamic'
fixed_winrate_threshold = 0.03
forecast_flat_stake = 10
forecast_results = []

# ✅ Simulation
for race_id, race_df in data.groupby('Race_ID', sort=False):
    full_field_size = len(race_df)
    if not ((4 <= full_field_size <= 6) or (full_field_size >= 41)):
        continue

    full_race = race_df.copy()
    full_race['Field_Size'] = full_field_size
    stake_pool = bankroll * bankroll_perc

    # Filter top 2 predicted horses
    top_two = full_race[full_race['Predicted_Rank'].isin([1, 2])].sort_values('Predicted_Rank')
    if len(top_two) != 2:
        continue

    b = top_two['Odds_To_Use'] - 1
    p = top_two['Predicted_Win_Probability']
    q = 1 - p
    top_two['Kelly_Fraction'] = ((b * p) - q) / b

    if winrate_filter_type == 'dynamic':
        top_two['Winrate_Threshold'] = 1 / full_field_size
    elif winrate_filter_type == 'fixed':
        top_two['Winrate_Threshold'] = fixed_winrate_threshold
    else:
        top_two['Winrate_Threshold'] = 0

    # Apply filters
    top_two['Valid'] = (
        (top_two['Kelly_Fraction'] > min_kelly) &
        (top_two['Expected_Value'] > min_ev) &
        (top_two['Odds_To_Use'] <= max_odds) &
        (top_two['Predicted_Win_Probability'] > top_two['Winrate_Threshold'])
    )

    if top_two['Valid'].sum() < 2:
        continue  # Both horses must qualify

    h1, h2 = top_two.iloc[0], top_two.iloc[1]
    o1, o2 = h1['Odds_To_Use'], h2['Odds_To_Use']
    p1, p2 = h1['Predicted_Win_Probability'], h2['Predicted_Win_Probability']

    s1 = h1['Kelly_Fraction'] * stake_pool
    s2 = h2['Kelly_Fraction'] * stake_pool
    total_stake = s1 + s2
    if total_stake > stake_pool:
        scale = stake_pool / total_stake
        s1 *= scale
        s2 *= scale
        total_stake = stake_pool

    # Calculate forecast metrics
    forecast_odds = o1 * o2
    forecast_prob = p1 * (p2 / (1 - p1))
    forecast_ev = (forecast_prob * (forecast_odds - 1)) - (1 - forecast_prob)

    # Determine forecast outcome
    actual_1st = full_race[full_race['Place'] == 1]['Horse'].values
    actual_2nd = full_race[full_race['Place'] == 2]['Horse'].values
    is_win = (
        len(actual_1st) > 0 and len(actual_2nd) > 0 and
        actual_1st[0] == h1['Horse'] and actual_2nd[0] == h2['Horse']
    )
    forecast_return = (forecast_odds - 1) * total_stake if is_win else -total_stake

    forecast_results.append({
        'Race_ID': race_id,
        'Horse_1': h1['Horse'],
        'Horse_2': h2['Horse'],
        'Odds_1': o1,
        'Odds_2': o2,
        'Stake_1': s1,
        'Stake_2': s2,
        'Total_Stake': total_stake,
        'Forecast_Odds': forecast_odds,
        'Forecast_Prob': forecast_prob,
        'Forecast_EV': forecast_ev,
        'Result': 'Win' if is_win else 'Loss',
        'Return': forecast_return,
        'R_Multiple': forecast_return / total_stake if total_stake > 0 else 0,
    })

# ✅ Save forecast results
forecast_df = pd.DataFrame(forecast_results)
forecast_df.to_excel('betting_simulation/forecast_only_results23-25.xlsx', index=False)

# ✅ Summary
if not forecast_df.empty:
    print("\n📊 Forecast Bets Summary:")
    print("✅ Total Forecast Bets:", len(forecast_df))
    print("✅ Total Forecast Profit: ${:.2f}".format(forecast_df['Return'].sum()))
    print("✅ Average R-Multiple: {:.4f}".format(forecast_df['R_Multiple'].mean()))
    print("✅ Win Rate: {:.1f}%".format((forecast_df['Result'] == 'Win').mean() * 100))
else:
    print("\n📭 No forecast bets placed.")
