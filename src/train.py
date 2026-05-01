import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def train_model():
    print("Loading processed data...")
    data_path = os.path.join('data', 'processed', 'train_cleaned.csv')
    df = pd.read_csv(data_path)
    X = df.drop(columns=['unit_number', 'RUL'])
    y = df['RUL']
    
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training XGBoost Regressor (This might take a few seconds)...")
    model = xgb.XGBRegressor(
        n_estimators=100,      
        learning_rate=0.1,    
        max_depth=5,         
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    predictions = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f"\n==========================================")
    print(f"   Validation RMSE: {rmse:.2f} cycles")
    print(f"==========================================\n")
    
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, 'xgboost_rul_model.json')
    print(f"Saving model to {model_path}...")
    model.save_model(model_path)
    print("Model saved successfully! Ready for production.")

if __name__ == "__main__":
    train_model()