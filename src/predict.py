import pandas as pd
import xgboost as xgb
import os

COLUMNS = ['unit_number', 'time_cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]

DROPPED_FEATURES = ['op_setting_3', 'sensor_1', 'sensor_10', 'sensor_18', 'sensor_19']

def run_inference():
    print("Loading saved XGBoost model...")
    model_path = os.path.join('models', 'xgboost_rul_model.json')
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    print("Loading test data and ground truth (answer key)...")
    test_path = os.path.join('data', 'raw', 'test_FD001.txt')
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMNS)
    
    truth_path = os.path.join('data', 'raw', 'RUL_FD001.txt')
    truth_df = pd.read_csv(truth_path, sep=r'\s+', header=None, names=['True_RUL'])
    
    engine_1_data = test_df[test_df['unit_number'] == 1]
    last_cycle_data = engine_1_data.iloc[[-1]].copy() 
    
    features = last_cycle_data.drop(columns=['unit_number'] + DROPPED_FEATURES)
    
    predicted_rul = model.predict(features)[0]
    
    actual_rul = truth_df.iloc[0]['True_RUL']
    
    print("\n============================================")
    print("   REAL-WORLD INFERENCE TEST (ENGINE 1)")
    print("============================================")
    print(f"Predicted Remaining Life: {predicted_rul:.0f} cycles")
    print(f"Actual Remaining Life:    {actual_rul} cycles")
    print(f"Off By:                   {abs(predicted_rul - actual_rul):.0f} cycles")
    print("============================================\n")

if __name__ == "__main__":
    run_inference()