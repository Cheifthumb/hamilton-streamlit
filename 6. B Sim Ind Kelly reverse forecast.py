import pandas as pd

# ✅ Load data
data = pd.read_csv('Betting_Simulation/predicted_win_probabilities.24-25.csv')
data['Place'] = pd.to_numeric(data['Place'], errors='coerce')
data = data.sort_values(by=['Date of Race', 'Time']).reset_index(drop=True)
data['Race_ID'] = data['Date of Race'].astype(str) + "_" + data['Time'].astype(str)

# ✅ Normalize predictions
data['Predicted_Rank'] = data.groupby('Race_ID')['Predicted_Win_Probability'].rank(method='first', ascending=False)
data['Odds_To_Use'] = data['Industry SP']
data['Predicted_Win_Probability'] = data.groupby('Race_ID')['Predicted_Win_Probability'].transform(lambda x: x / x.sum())
data['Expected_Value'] = (data['Predicted_Win_Probability'] * (data['Odds_To_Use'] - 1)) - (1 - data['Predicted_Win_Probability'])

# ✅ Settings
bankroll = 10000
bankroll_perc = 0.1
min_ev = -5
min_kelly = -0.5
max_odds = 100.0
winrate_filter_type = 'none'  # or 'fixed', 'dynamic'
fixed_winrate_threshold = 0.03
reverse_forecast_total_stake = 50

reverse_results = []

# ✅ Loop over races
for race_id, race_df in data.groupby('Race_ID', sort=False):
    full_field_size = len(race_df)
    if not ((4 <= full_field_size <= 6) or (full_field_size >= 41)):
        continue

    full_race = race_df.copy()
    full_race['Field_Size'] = full_field_size
    stake_pool = bankroll * bankroll_perc

    # Get predicted rank 1 and 2
    top_two = full_race[full_race['Predicted_Rank'].isin([1, 2])].sort_values('Predicted_Rank')
    if len(top_two) != 2:
        continue

    # Calculate Kelly fraction and winrate filters
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

    top_two['Valid'] = (
        (top_two['Kelly_Fraction'] > min_kelly) &
        (top_two['Expected_Value'] > min_ev) &
        (top_two['Odds_To_Use'] <= max_odds) &
        (top_two['Predicted_Win_Probability'] > top_two['Winrate_Threshold'])
    )

    if top_two['Valid'].sum() < 2:
        continue

    h1, h2 = top_two.iloc[0], top_two.iloc[1]
    o1, o2 = h1['Odds_To_Use'], h2['Odds_To_Use']
    p1, p2 = h1['Predicted_Win_Probability'], h2['Predicted_Win_Probability']

    # Forecast A→B
    prob_ab = p1 * (p2 / (1 - p1))
    odds_ab = o1 * o2
    ev_ab = (prob_ab * (odds_ab - 1)) - (1 - prob_ab)

    # Forecast B→A
    prob_ba = p2 * (p1 / (1 - p2))
    odds_ba = o2 * o1
    ev_ba = (prob_ba * (odds_ba - 1)) - (1 - prob_ba)

    # Determine actual outcome
    actual_1st = full_race[full_race['Place'] == 1]['Horse'].values
    actual_2nd = full_race[full_race['Place'] == 2]['Horse'].values

    is_ab_win = len(actual_1st) > 0 and len(actual_2nd) > 0 and actual_1st[0] == h1['Horse'] and actual_2nd[0] == h2['Horse']
    is_ba_win = len(actual_1st) > 0 and len(actual_2nd) > 0 and actual_1st[0] == h2['Horse'] and actual_2nd[0] == h1['Horse']

    stake_each = reverse_forecast_total_stake / 2
    ret_ab = (odds_ab - 1) * stake_each if is_ab_win else -stake_each
    ret_ba = (odds_ba - 1) * stake_each if is_ba_win else -stake_each
    total_return = ret_ab + ret_ba

    reverse_results.append({
        'Race_ID': race_id,
        'Horse_A': h1['Horse'],
        'Horse_B': h2['Horse'],
        'Odds_A': o1,
        'Odds_B': o2,
        'Stake_AB': stake_each,
        'Stake_BA': stake_each,
        'Prob_AB': prob_ab,
        'Prob_BA': prob_ba,
        'EV_AB': ev_ab,
        'EV_BA': ev_ba,
        'Forecast_Odds_AB': odds_ab,
        'Forecast_Odds_BA': odds_ba,
        'Result_AB': 'Win' if is_ab_win else 'Loss',
        'Result_BA': 'Win' if is_ba_win else 'Loss',
        'Return_AB': ret_ab,
        'Return_BA': ret_ba,
        'Total_Stake': reverse_forecast_total_stake,
        'Total_Return': total_return,
        'R_Multiple': total_return / reverse_forecast_total_stake
    })

# ✅ Save results
reverse_df = pd.DataFrame(reverse_results)
# Add field size info before saving
field_sizes = data.groupby('Race_ID')['Horse'].count()
reverse_df['Field_Size'] = reverse_df['Race_ID'].map(field_sizes)

# Save to Excel
reverse_df.to_excel('betting_simulation/reverse_forecast_results_23-25.xlsx', index=False)


# ✅ Summary
if not reverse_df.empty:
    print("\n📊 Reverse Forecast Bets Summary:")
    print("✅ Total Reverse Forecasts:", len(reverse_df))
    print("✅ Total Profit: ${:.2f}".format(reverse_df['Total_Return'].sum()))
    print("✅ Avg R-Multiple: {:.4f}".format(reverse_df['R_Multiple'].mean()))
    print("✅ Win AB: {} | Win BA: {}".format(
        (reverse_df['Result_AB'] == 'Win').sum(),
        (reverse_df['Result_BA'] == 'Win').sum()
    ))
    print("✅ Win Rate (either leg): {:.1f}%".format(
        ((reverse_df['Result_AB'] == 'Win') | (reverse_df['Result_BA'] == 'Win')).mean() * 100
    ))
else:
    print("\n📭 No reverse forecast bets placed.")

# ✅ Optional: Breakdown by field size bin
if not reverse_df.empty:
    print("\n📦 Reverse Forecast Return by Field Size Bin:")

    # Calculate field size for each race from the original data
    field_sizes = data.groupby('Race_ID')['Horse'].count()
    reverse_df['Field_Size'] = reverse_df['Race_ID'].map(field_sizes)

    # Bin field sizes
    field_bins = [0, 4,5,6,7, 8,9,10,11, 12, 20, 100]
    bin_labels = ['≤4','5', '6','7','8', '9','10','11','12', '13-20', '21+']
    reverse_df['Field_Bin'] = pd.cut(reverse_df['Field_Size'], bins=field_bins, labels=bin_labels, right=True)

    # Compute win (AB or BA) per row
    reverse_df['Either_Win'] = (reverse_df['Result_AB'] == 'Win') | (reverse_df['Result_BA'] == 'Win')

    # Aggregate by bin
    bin_summary = reverse_df.groupby('Field_Bin').agg(
        Num_Bets=('R_Multiple', 'count'),
        Total_Return=('Total_Return', 'sum'),
        Avg_R_Multiple=('R_Multiple', 'mean'),
        Win_Rate=('Either_Win', 'mean')
    ).reset_index()

    # Format win rate as percentage
    bin_summary['Win_Rate'] = (bin_summary['Win_Rate'] * 100).round(1).astype(str) + '%'

    print(bin_summary.to_string(index=False))

