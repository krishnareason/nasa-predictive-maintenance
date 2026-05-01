import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data():

    columns = ['unit_number', 'time_cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    columns += [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv('data/raw/train_FD004.txt', sep='\s+', header=None, names=columns)
    
    rul = pd.DataFrame(df.groupby('unit_number')['time_cycle'].max()).reset_index()
    rul.columns = ['unit_number', 'max_cycle']
    df = df.merge(rul, on=['unit_number'], how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def main():
    print("🚀 Starting Advanced FD004 Training Pipeline...")
    df = load_data()
    
    op_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    
    print("🧠 Clustering Flight Regimes using K-Means...")
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df['flight_regime'] = kmeans.fit_predict(df[op_settings])
    
    print("⚖️ Normalizing sensors based on flight altitude/regime...")
    features_to_scale = op_settings + sensor_cols
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    print("⚙️ Training XGBoost Regression Model on FD004...")
    features = features_to_scale + ['flight_regime']
    X = df[features]
    y = df['RUL']
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    print("💾 Saving Model, Scaler, and KMeans to /models folder...")
    os.makedirs('models', exist_ok=True)
    model.save_model(os.path.join('models', 'xgboost_fd004_model.json'))
    joblib.dump(kmeans, os.path.join('models', 'kmeans_fd004.joblib'))
    joblib.dump(scaler, os.path.join('models', 'scaler_fd004.joblib'))
    
    print("✅ Training Complete! Advanced FD004 Model is ready.")

if __name__ == "__main__":
    main()