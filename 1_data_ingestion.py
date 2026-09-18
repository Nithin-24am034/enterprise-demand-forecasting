import os
import pandas as pd 
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("DATABASE_URL")

if not db_url:
    raise ValueError("DATABASE_URL not found. Ensure your .env file is configured correctly.")

if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://","postgresql://",1)

print("Connecting to Neon PostgreSQL")
engine = create_engine(db_url)

print("Loading train.csv...")
df = pd.read_csv("train.csv")
df["date"] = pd.to_datetime(df["date"])

print(f"Total rows to upload: {len(df):,}")
print("Uploading data to Neon (uploading in chunks to prevent timeouts)...")

df.to_sql("sales_data",engine, if_exists="replace", index=False, chunksize=10000)

print("\nVerifying data insertion...")
test_query = pd.read_sql_query("""
SELECT store, item, COUNT(*) as record_count, SUM(sales) as total_units_sold
FROM sales_data
WHERE store = 1 AND item = 1
GROUP  BY store, item
""", engine)

print(test_query)
print("\nStep 1 Complete: Dataset successfully stored in Neon Postgres!")
