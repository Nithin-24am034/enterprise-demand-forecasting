import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 1. Connect to the cloud database
load_dotenv()
db_url = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://", 1)
engine = create_engine(db_url)

print("Fetching data and applying features...")
query = """
    SELECT date, store_id, item_id, sales, price, promo, weekday, month
    FROM sales_data
    WHERE store_id = 'store_1' AND item_id = 'item_1'
    ORDER BY date ASC
"""
df = pd.read_sql_query(query, engine)
df['date'] = pd.to_datetime(df['date'])

# 2. Re-apply Feature Engineering
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['sales_lag_7'] = df['sales'].shift(7)
df['sales_rolling_avg_7'] = df['sales'].rolling(window=7).mean()
df = df.dropna()

# 3. Train-Test Split (Chronological)
features = ['day_of_week', 'month', 'is_weekend', 'sales_lag_7', 'sales_rolling_avg_7', 'price', 'promo']
target = 'sales'

X = df[features]
y = df[target]

# Split 80% for training, 20% for testing
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 4. Model Training
print("Training XGBoost Model...")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions & MAPE Evaluation
print("Evaluating model...")
predictions = model.predict(X_test)

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero issues
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

mape = calculate_mape(y_test, predictions)

print(f"\n✅ Model Training Complete!")
print(f"📉 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 6. Save the trained model
print("\nSaving model to disk...")
model_filename = 'xgboost_demand_model.pkl'
joblib.dump(model, model_filename)
print(f"✅ Model saved successfully as {model_filename}")