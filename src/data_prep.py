import pandas as pd
import os

COLUMNS = ['unit_number', 'time_cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]

def load_data(file_path):
    """Loads the NASA turbofan dataset and assigns proper column names."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=COLUMNS)
    return df

def add_rul(df):
    """Calculates and adds the Remaining Useful Life (RUL)."""
    print("Engineering Remaining Useful Life (RUL)...")
    rul_data = df.groupby('unit_number')['time_cycle'].max().reset_index()
    rul_data.columns = ['unit_number', 'max_cycle']
    df = df.merge(rul_data, on=['unit_number'], how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycle']
    df = df.drop('max_cycle', axis=1)
    return df

def clean_data(df):
    """Removes sensors that have zero variance (constant values)."""
    print("Cleaning data (Dropping constant sensors)...")
    cols_to_check = [c for c in df.columns if 'sensor' in c or 'op_setting' in c]
    
    cols_to_drop = []
    for col in cols_to_check:
        if df[col].std() == 0:
            cols_to_drop.append(col)
            
    print(f"-> Useless sensors dropped: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    
    return df

if __name__ == "__main__":
    raw_data_path = os.path.join('data', 'raw', 'train_FD001.txt')
    processed_data_path = os.path.join('data', 'processed', 'train_cleaned.csv')
    
    train_df = load_data(raw_data_path)
    train_df = add_rul(train_df)
    train_df = clean_data(train_df)
    
    print(f"Saving processed data to {processed_data_path}...")
    train_df.to_csv(processed_data_path, index=False)
    
    print("\n--- Data Prep Complete! ---")
    print(f"Final data shape for training: {train_df.shape[0]} rows, {train_df.shape[1]} columns.")