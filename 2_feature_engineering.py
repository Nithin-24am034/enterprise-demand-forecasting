import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 1. Connect to the cloud database
load_dotenv()
db_url = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://", 1)
engine = create_engine(db_url)

# 2. Extract a single time series to build our features
print("Extracting data for store_1 and item_1...")
query = """
    SELECT date, store_id, item_id, sales, price, promo
    FROM sales_data
    WHERE store_id = 'store_1' AND item_id = 'item_1'
    ORDER BY date ASC
"""
df = pd.read_sql_query(query, engine)
df['date'] = pd.to_datetime(df['date'])

# 3. Time-based Features (Teaching the model about calendars)
print("Generating time-based features...")
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 4. Lag & Rolling Features (Teaching the model about past momentum)
print("Generating lag and rolling features...")
# What were the sales exactly one week ago?
df['sales_lag_7'] = df['sales'].shift(7) 
# What is the average sales trend over the last 7 days?
df['sales_rolling_avg_7'] = df['sales'].rolling(window=7).mean()

# The first 7 days won't have historical lag data, so we drop those empty rows
df = df.dropna()

print("\n✅ Feature Engineering Complete. Here is a preview of your ML-ready data:")
print(df[['date', 'sales', 'sales_lag_7', 'sales_rolling_avg_7', 'is_weekend']].head(10))