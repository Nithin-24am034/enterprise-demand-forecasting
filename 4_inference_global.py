import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import timedelta

print("Loading Global XGBoost model...")
# 1. Load the new global model
model = joblib.load('xgboost_global_demand_model.pkl')

load_dotenv()
db_url = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://", 1)
engine = create_engine(db_url)

# 2. Define exactly what we want to predict (Change these to test different products!)
target_store = 'store_3'
target_item = 'item_50'

print(f"Fetching recent historical data for {target_store}, {target_item} from Neon...")
query = f"""
    SELECT date, sales, price, promo
    FROM sales_data
    WHERE store_id = '{target_store}' AND item_id = '{target_item}'
    ORDER BY date DESC
    LIMIT 7
"""
df_recent = pd.read_sql_query(query, engine)
df_recent['date'] = pd.to_datetime(df_recent['date'])
df_recent = df_recent.sort_values('date', ascending=True).reset_index(drop=True)

# 3. Prepare Tomorrow's Features
last_date = df_recent['date'].iloc[-1]
target_date = last_date + timedelta(days=1)
print(f"\nGenerating forecast for: {target_date.date()}")

sales_lag_7 = df_recent['sales'].iloc[0]
sales_rolling_avg_7 = df_recent['sales'].mean()

# 4. Construct the inference row (Now including categorical IDs)
future_features = pd.DataFrame([{
    'store_id': target_store,
    'item_id': target_item,
    'day_of_week': target_date.dayofweek,
    'month': target_date.month,
    'is_weekend': 1 if target_date.dayofweek in [5, 6] else 0,
    'sales_lag_7': sales_lag_7,
    'sales_rolling_avg_7': sales_rolling_avg_7,
    'price': df_recent['price'].iloc[-1],
    'promo': df_recent['promo'].iloc[-1]
}])

# Convert to categories to match the training data format
future_features['store_id'] = future_features['store_id'].astype('category')
future_features['item_id'] = future_features['item_id'].astype('category')

# 5. Make the Prediction
prediction = model.predict(future_features)[0]

print("-" * 30)
print(f"🔮 Predicted Sales Volume: {int(prediction)} units")
print("-" * 30)