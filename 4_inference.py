import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import timedelta

# 1. Load the trained "brain"
print("Loading saved XGBoost model...")
model = joblib.load('xgboost_demand_model.pkl')

# 2. Connect to the database
load_dotenv()
db_url = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://", 1)
engine = create_engine(db_url)

# 3. Fetch ONLY the most recent 7 days of historical data for context
print("Fetching recent historical data from Neon...")
query = """
    SELECT date, sales, price, promo
    FROM sales_data
    WHERE store_id = 'store_1' AND item_id = 'item_1'
    ORDER BY date DESC
    LIMIT 7
"""
df_recent = pd.read_sql_query(query, engine)
df_recent['date'] = pd.to_datetime(df_recent['date'])
# Reverse the sort so the oldest of the 7 days is first, newest is last
df_recent = df_recent.sort_values('date', ascending=True).reset_index(drop=True)

# 4. Prepare "Tomorrow's" Features
last_date = df_recent['date'].iloc[-1]
target_date = last_date + timedelta(days=1)
print(f"\nGenerating forecast for: {target_date.date()}")

# Calculate sequential features dynamically based on the recent data
sales_lag_7 = df_recent['sales'].iloc[0]  # Sales from exactly 7 days ago
sales_rolling_avg_7 = df_recent['sales'].mean()  # Average of the last 7 days

# Time features for tomorrow
day_of_week = target_date.dayofweek
month = target_date.month
is_weekend = 1 if day_of_week in [5, 6] else 0

# Carry over the most recent price and promo status
recent_price = df_recent['price'].iloc[-1]
recent_promo = df_recent['promo'].iloc[-1]

# Construct the single row of data the model expects
future_features = pd.DataFrame([{
    'day_of_week': day_of_week,
    'month': month,
    'is_weekend': is_weekend,
    'sales_lag_7': sales_lag_7,
    'sales_rolling_avg_7': sales_rolling_avg_7,
    'price': recent_price,
    'promo': recent_promo
}])

# 5. Make the Prediction
prediction = model.predict(future_features)[0]

print("-" * 30)
print(f"🔮 Predicted Sales Volume: {int(prediction)} units")
print("-" * 30)