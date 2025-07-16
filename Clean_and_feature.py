import pandas as pd
import numpy as np
import os

def clean_and_engineer(file_path, output_name="model_data"):
    cleaned_folder = "cleaned_data"
    featured_folder = "featured_data"

    os.makedirs(cleaned_folder, exist_ok=True)
    os.makedirs(featured_folder, exist_ok=True)

    # Load input
    df = pd.read_excel(file_path)

    # === CLEANING ===
    required_cols = ['Date of Race', 'Horse', 'Industry SP']
    dropped_rows = pd.DataFrame()
    reasons = []

    for col in required_cols:
        if col in df.columns:
            missing = df[df[col].isnull()]
            if not missing.empty:
                dropped_rows = pd.concat([dropped_rows, missing])
                reasons.extend([f"Missing {col}"] * len(missing))
            df = df[df[col].notnull()]

    if 'Date of Race' in df.columns:
        df['Date of Race'] = pd.to_datetime(df['Date of Race'], errors='coerce')
        bad_dates = df[df['Date of Race'].isnull()]
        if not bad_dates.empty:
            dropped_rows = pd.concat([dropped_rows, bad_dates])
            reasons.extend(["Invalid date"] * len(bad_dates))
        df = df[df['Date of Race'].notnull()]

    if 'Industry SP' in df.columns:
        df['Industry SP'] = pd.to_numeric(df['Industry SP'], errors='coerce')
        bad_sp = df[df['Industry SP'].isnull()]
        if not bad_sp.empty:
            dropped_rows = pd.concat([dropped_rows, bad_sp])
            reasons.extend(["Invalid odds"] * len(bad_sp))
        df = df[df['Industry SP'].notnull()]

    # Save cleaned
    cleaned_file = os.path.join(cleaned_folder, f"{output_name}_CLEAN.xlsx")
    df.to_excel(cleaned_file, index=False)
    print(f"✅ Cleaned data saved to {cleaned_file} with {len(df)} rows.")

    if not dropped_rows.empty:
        dropped_rows['Reason Removed'] = reasons
        audit_file = os.path.join(cleaned_folder, f"{output_name}_AUDIT.xlsx")
        dropped_rows.to_excel(audit_file, index=False)
        print(f"⚠️ Audit log saved to {audit_file} with {len(dropped_rows)} removed rows.")
    else:
        print("✅ No rows removed during cleaning.")

    # === FEATURE ENGINEERING ===
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
        df['Days Since Last Run'] = df.groupby('Horse')['Date of Race'].diff().dt.days

    if 'Industry SP' in df.columns:
        df['Log Industry SP'] = df['Industry SP'].apply(lambda x: pd.NA if pd.isna(x) or x <= 0 else np.log(x))

    if 'SP Fav' in df.columns:
        df['Is Favourite'] = df['SP Fav'].apply(lambda x: 1 if str(x).strip().lower() == 'fav' else 0)

    if 'Industry SP' in df.columns and 'Date of Race' in df.columns and 'Track' in df.columns:
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

    # Save featured
    featured_file = os.path.join(featured_folder, f"{output_name}_FEATURED.xlsx")
    df.to_excel(featured_file, index=False)
    print(f"✨ Features saved to {featured_file} with {len(df)} rows.")

    return featured_file

if __name__ == "__main__":
    # 👇 Update with your file and desired name
    file_path = "Years/02-25.xlsx"
    output_name = "02-25"
    clean_and_engineer(file_path, output_name=output_name)
