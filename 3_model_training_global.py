import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

print("1. Loading all 4.5 Million rows locally to bypass cloud network delay...")
df = pd.read_csv('train.csv')
df['date'] = pd.to_datetime(df['date'])

# Sort chronologically to ensure our time math is perfect
df = df.sort_values(['store_id', 'item_id', 'date'])

print("2. Generating grouped sequential features (Preventing Data Leakage)...")
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# CRITICAL: We group by store and item BEFORE shifting, so history doesn't bleed across products
df['sales_lag_7'] = df.groupby(['store_id', 'item_id'])['sales'].shift(7)
df['sales_rolling_avg_7'] = df.groupby(['store_id', 'item_id'])['sales'].transform(lambda x: x.rolling(window=7).mean())

# Drop the first 7 days of every item that lack historical lag data
df = df.dropna()

print("3. Formatting Categorical Features...")
# Convert text IDs into XGBoost-friendly categories to remove bias
df['store_id'] = df['store_id'].astype('category')
df['item_id'] = df['item_id'].astype('category')

features = ['store_id', 'item_id', 'day_of_week', 'month', 'is_weekend', 'sales_lag_7', 'sales_rolling_avg_7', 'price', 'promo']
target = 'sales'

# Chronological Train-Test Split (Train on everything except the last 90 days)
split_date = df['date'].max() - pd.Timedelta(days=90)
train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train, y_train = df[train_mask][features], df[train_mask][target]
X_test, y_test = df[test_mask][features], df[test_mask][target]

print("4. Training Global XGBoost Model (This will take a few minutes)...")
# enable_categorical=True allows the model to process store_id and item_id fairly
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, enable_categorical=True)
model.fit(X_train, y_train)

print("5. Evaluating Global Model...")
predictions = model.predict(X_test)

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

mape = calculate_mape(y_test, predictions)
print(f"📉 Global Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

print("\n6. Saving global model to disk...")
model_filename = 'xgboost_global_demand_model.pkl'
joblib.dump(model, model_filename)
print(f"✅ Global Model saved successfully as {model_filename}")