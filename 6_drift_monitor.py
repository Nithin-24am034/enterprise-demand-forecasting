import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import subprocess

print("Starting Model Drift Monitor...")

# 1. Connect to the Cloud Database
load_dotenv()
db_url = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://", 1)
engine = create_engine(db_url)

# 2. Fetch Predictions vs. Actuals
# This joins the predictions your API logged with the actual sales that happened that day
query = """
    SELECT 
        p.store_id, 
        p.item_id, 
        p.forecast_date, 
        p.predicted_sales, 
        s.sales as actual_sales
    FROM prediction_logs p
    JOIN sales_data s 
      ON p.store_id = s.store_id 
     AND p.item_id = s.item_id 
     AND p.forecast_date = s.date
"""
df_eval = pd.read_sql_query(query, engine)

if df_eval.empty:
    print("No matching actual sales data found for recent predictions.")
    print("Waiting for new actuals to arrive in the database...")
    exit()

# 3. Calculate Live Production MAPE
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if sum(non_zero_mask) == 0: return 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

current_mape = calculate_mape(df_eval['actual_sales'], df_eval['predicted_sales'])
print(f"📊 Current Production MAPE: {current_mape:.2f}%")

# 4. Trigger Retraining if Drift is Detected
DRIFT_THRESHOLD = 15.0  # If the error exceeds 15%, the model is degrading

if current_mape > DRIFT_THRESHOLD:
    print(f"⚠️ ALERT: Model drift detected (MAPE {current_mape:.2f}% > {DRIFT_THRESHOLD}%).")
    print("🔄 Initiating automated retraining pipeline...")
    
    try:
        # MLOps: Automatically execute the training script to build a fresh model
        subprocess.run(["python", "3_model_training_global.py"], check=True)
        print("✅ Retraining complete. New global model deployed to production.")
        
    except subprocess.CalledProcessError:
        print("❌ ERROR: Automated retraining failed. Check the training script.")
else:
    print("✅ Model is healthy. No retraining required.")