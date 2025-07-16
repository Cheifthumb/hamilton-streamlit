import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load and prepare data
data = pd.read_csv('Betting_Simulation/predicted_win_probabilities.23-25.csv')
data['Place'] = pd.to_numeric(data['Place'], errors='coerce')
data = data.sort_values(by=['Date of Race', 'Time']).reset_index(drop=True)
data['Race_ID'] = data['Date of Race'].astype(str) + "_" + data['Time'].astype(str)

# ✅ Rank horses by predicted win probability
data['Predicted_Rank'] = data.groupby('Race_ID')['Predicted_Win_Probability'].rank(method='first', ascending=False)
data['Top_3_Flag'] = data['Predicted_Rank'] <= 3

data['Odds_To_Use'] = data['Industry SP']
data['Predicted_Win_Probability'] = data.groupby('Race_ID')['Predicted_Win_Probability'].transform(lambda x: x / x.sum())
data['Expected_Value'] = (data['Predicted_Win_Probability'] * (data['Odds_To_Use'] - 1)) - (1 - data['Predicted_Win_Probability'])

# ✅ Simulation settings
initial_bankroll = 10000
current_bankroll = initial_bankroll
bankroll_perc = 0.01
min_ev_threshold = 0.10
min_kelly_fraction = 0.01
max_odds_threshold = 100.0
min_odds_threshold = 1.5  # ✅ New minimum odds threshold

# ✅ Stake mode: 'kelly', 'fixed', or 'winrate'
stake_mode = 'winrate'
fixed_stake_perc = 0.01

# ✅ Win rate filter settings
winrate_filter_type = 'none'  # Options: 'none', 'fixed', 'dynamic'
fixed_winrate_threshold = 0.03

# ✅ Rank filter: Only include predicted rank 1, 2, or 3 horses
allowed_predicted_ranks = [1, 2]

# ✅ Track filter
track_filter = None
if track_filter is not None:
    data = data[data['Track'].isin(track_filter)]
print(f"🏇 Track Filter: {track_filter if track_filter else 'All tracks'}")

updated_rows, rejected_rows = [], []

# ✅ Race-by-race simulation
for race_id, race_df in data.groupby('Race_ID', sort=False):
    race_df = race_df.copy()
    full_field_size = len(race_df)
    race_df['Field_Size'] = full_field_size

    race_df = race_df[race_df['Predicted_Rank'].isin(allowed_predicted_ranks)]

    if race_df.empty:
        print(f"⏩ Skipping Race {race_id} (No runners match rank filter)")
        continue

    if not ((4 <= full_field_size <= 6) or (full_field_size >= 41)):
        print(f"⏩ Skipping Race {race_id} (Field size = {full_field_size})")
        continue

    print(f"✅ Processing Race {race_id} with {full_field_size} horses")

    stake_pool = current_bankroll * bankroll_perc
    b = race_df['Odds_To_Use'] - 1
    p = race_df['Predicted_Win_Probability']
    q = 1 - p
    race_df['Kelly_Fraction'] = ((b * p) - q) / b

    race_df['Reject_Reason'] = ''
    race_df.loc[race_df['Kelly_Fraction'] <= min_kelly_fraction, 'Reject_Reason'] += 'kelly_low|'
    race_df.loc[race_df['Expected_Value'] <= min_ev_threshold, 'Reject_Reason'] += 'ev_low|'
    race_df.loc[race_df['Odds_To_Use'] > max_odds_threshold, 'Reject_Reason'] += 'odds_high|'
    race_df.loc[race_df['Odds_To_Use'] < min_odds_threshold, 'Reject_Reason'] += 'odds_low|'

    if winrate_filter_type == 'dynamic':
        race_df['Winrate_Threshold'] = 1 / full_field_size
    elif winrate_filter_type == 'fixed':
        race_df['Winrate_Threshold'] = fixed_winrate_threshold
    else:
        race_df['Winrate_Threshold'] = 0

    race_df.loc[race_df['Predicted_Win_Probability'] <= race_df['Winrate_Threshold'], 'Reject_Reason'] += 'winrate_low|'

    race_df['Bet_Placed'] = (
        (race_df['Kelly_Fraction'] > min_kelly_fraction) &
        (race_df['Expected_Value'] > min_ev_threshold) &
        (race_df['Odds_To_Use'] >= min_odds_threshold) &
        (race_df['Odds_To_Use'] <= max_odds_threshold) &
        (race_df['Predicted_Win_Probability'] > race_df['Winrate_Threshold'])
    )

    # ✅ Calculate stake based on selected mode
    race_df['Stake'] = 0
    if stake_mode == 'kelly':
        race_df.loc[race_df['Bet_Placed'], 'Stake'] = race_df.loc[race_df['Bet_Placed'], 'Kelly_Fraction'] * stake_pool
    elif stake_mode == 'fixed':
        qualifying_bets = race_df['Bet_Placed'].sum()
        if qualifying_bets > 0:
            stake_per_bet = stake_pool / qualifying_bets
            race_df.loc[race_df['Bet_Placed'], 'Stake'] = stake_per_bet
    elif stake_mode == 'winrate':
        probs = race_df.loc[race_df['Bet_Placed'], 'Predicted_Win_Probability']
        if not probs.empty:
            race_df.loc[race_df['Bet_Placed'], 'Stake'] = (probs / probs.sum()) * stake_pool

    total_stake = race_df['Stake'].sum()
    if total_stake > stake_pool and total_stake > 0:
        race_df.loc[race_df['Bet_Placed'], 'Stake'] *= stake_pool / total_stake

    race_df['Actual_Result'] = (race_df['Place'] == 1).astype(int)
    race_df['Bet_Return'] = 0
    win_mask = race_df['Bet_Placed'] & (race_df['Actual_Result'] == 1)
    lose_mask = race_df['Bet_Placed'] & (race_df['Actual_Result'] == 0)

    race_df.loc[win_mask, 'Bet_Return'] = (race_df.loc[win_mask, 'Odds_To_Use'] - 1) * race_df.loc[win_mask, 'Stake']
    race_df.loc[lose_mask, 'Bet_Return'] = -race_df.loc[lose_mask, 'Stake']

    current_bankroll += race_df['Bet_Return'].sum()
    race_df['Bankroll_After_Race'] = current_bankroll

    updated_rows.append(race_df)
    rejected_rows.append(race_df[~race_df['Bet_Placed']])

# ✅ Post-simulation
if updated_rows:
    data = pd.concat(updated_rows).reset_index(drop=True)
    rejected_data = pd.concat(rejected_rows).reset_index(drop=True)
    data['Max_Bankroll'] = data['Bankroll_After_Race'].cummax()
    data['Drawdown'] = data['Max_Bankroll'] - data['Bankroll_After_Race']
else:
    print("\n❌ No races met the filter criteria — no bets were placed.")
    data = pd.DataFrame()
    rejected_data = pd.DataFrame()

if not data.empty:
    data['R_Multiple'] = 0
    data.loc[data['Bet_Placed'] & (data['Stake'] > 0), 'R_Multiple'] = data['Bet_Return'] / data['Stake']

    total_bets = data['Bet_Placed'].sum()
    total_staked = data.loc[data['Bet_Placed'], 'Stake'].sum()
    total_profit = data['Bet_Return'].sum()
    final_bankroll = current_bankroll
    average_R = data.loc[data['Bet_Placed'], 'R_Multiple'].mean()
    total_winning_R = data.loc[data['R_Multiple'] > 0, 'R_Multiple'].sum()
    total_losing_R = data.loc[data['R_Multiple'] < 0, 'R_Multiple'].sum()
    max_drawdown = data['Drawdown'].max()

    print("\n✅ Total Bets Placed:", int(total_bets))
    print("✅ Total Amount Staked: ${:.2f}".format(total_staked))
    print("✅ Total Profit/Loss: ${:.2f}".format(total_profit))
    print("🏦 Final Bankroll: ${:.2f}".format(final_bankroll))
    print("✅ Avg R-Multiple: {:.4f}".format(average_R))
    print("📈 Total Winning R: {:.2f}".format(total_winning_R))
    print("📉 Total Losing R: {:.2f}".format(total_losing_R))
    print("📉 Max Drawdown: ${:.2f}".format(max_drawdown))

    data.to_csv('betting_simulation/betting_simulation_kelly_24-25.csv', index=False)
    rejected_data.to_csv('betting_simulation/rejected_bets.csv', index=False)

    def group_r_stats(df, group_col):
        df = df[df['Bet_Placed']]
        grouped = df.groupby(group_col)
        stats = grouped['R_Multiple'].agg(
            Total_Bets='count',
            Total_R='sum',
            Winning_R=lambda x: x[x > 0].sum(),
            Losing_R=lambda x: x[x < 0].sum()
        )
        stats['Win_Rate (%)'] = grouped['R_Multiple'].apply(lambda x: (x > 0).mean() * 100).round(1)
        return stats

    data['Odds_Bin'] = pd.cut(data['Odds_To_Use'], bins=[0, 5, 10, 15, 25, 50, 100])
    r_stats_odds = group_r_stats(data, 'Odds_Bin')
    print("\n📊 R-Metrics by Odds Bin:\n", r_stats_odds)

    def clean_class(val):
        try: return int(''.join(filter(str.isdigit, str(val))))
        except: return None

    data['Race_Class'] = data['Class'].apply(clean_class) if 'Class' in data.columns else None
    if data['Race_Class'].notna().any():
        r_stats_class = group_r_stats(data, 'Race_Class')
        print("\n📊 R-Metrics by Race Class:\n", r_stats_class)

    data['Field_Size_Bin'] = pd.cut(data['Field_Size'], bins=[0, 4, 5, 6, 7, 8, 9, 10, 13, 20], labels=['1–4', '5', '6', '7', '8', '9', '10', '11–13', '14+'])
    r_stats_field = group_r_stats(data, 'Field_Size_Bin')
    print("\n📊 R-Metrics by Field Size:\n", r_stats_field)

    if 'Track' in data.columns and data['Track'].notna().any():
        r_stats_track = group_r_stats(data, 'Track').sort_values('Total_R', ascending=False)
        print("\n📊 R-Metrics by Track:\n", r_stats_track)

    data['Race_DateTime'] = pd.to_datetime(data['Date of Race'] + ' ' + data['Time'])

    with pd.ExcelWriter('betting_simulation/grouped_r_metrics.xlsx', engine='openpyxl') as writer:
        r_stats_odds.to_excel(writer, sheet_name='By_Odds_Bin')
        if 'Race_Class' in data.columns and data['Race_Class'].notna().any():
            r_stats_class.to_excel(writer, sheet_name='By_Race_Class')
        if 'Field_Size_Bin' in data.columns:
            r_stats_field.to_excel(writer, sheet_name='By_Field_Size')
        if 'Track' in data.columns and data['Track'].notna().any():
            r_stats_track.to_excel(writer, sheet_name='By_Track')

    plt.figure(figsize=(12, 6))
    plt.plot(data['Race_DateTime'], data['Bankroll_After_Race'], label='Bankroll', marker='o', linewidth=1, markersize=2)
    plt.axhline(initial_bankroll, color='gray', linestyle='--', label='Starting Bankroll')
    plt.title('📈 Bankroll Over Time (R-Multiple Evaluation)')
    plt.xlabel('Date')
    plt.ylabel('Bankroll ($)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("📭 Skipping stats and plots — no simulation data generated.")
