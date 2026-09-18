from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from datetime import timedelta, datetime

app = FastAPI(title="Enterprise Demand Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model into server memory...")
model = joblib.load('xgboost_global_demand_model.pkl')

load_dotenv()
db_url = os.getenv("DATABASE_URL").replace("postgres://", "postgresql://", 1)
engine = create_engine(db_url)

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            log_id SERIAL PRIMARY KEY,
            store_id VARCHAR(50),
            item_id VARCHAR(50),
            forecast_date DATE,
            predicted_sales INT,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.commit()

# Updated Request Format: Now accepts a specific target date
class ForecastRequest(BaseModel):
    store_id: str
    item_id: str
    target_date: str 

@app.post("/predict")
def predict_demand(request: ForecastRequest):
    target_store = request.store_id
    target_item = request.item_id
    
    # Parse the user's custom date
    try:
        requested_date = datetime.strptime(request.target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    query = f"""
        SELECT date, sales, price, promo
        FROM sales_data
        WHERE store_id = '{target_store}' AND item_id = '{target_item}'
        ORDER BY date DESC
        LIMIT 7
    """
    df_recent = pd.read_sql_query(query, engine)
    if df_recent.empty:
        raise HTTPException(status_code=404, detail="Store/Item combination not found.")
        
    df_recent['date'] = pd.to_datetime(df_recent['date'])
    df_recent = df_recent.sort_values('date', ascending=True).reset_index(drop=True)
    
    current_date = df_recent['date'].iloc[-1].date()
    
    # Calculate how many days into the future we need to run the loop
    horizon = (requested_date - current_date).days
    
    if horizon <= 0:
        raise HTTPException(status_code=400, detail="Target date must be strictly in the future.")
        
    working_sales_history = df_recent['sales'].tolist()
    current_price = df_recent['price'].iloc[-1]
    current_promo = df_recent['promo'].iloc[-1]
    
    forecast_results = []
    loop_date = current_date
    
    # Run the recursive loop until we reach the custom date
    for i in range(horizon):
        loop_date = loop_date + timedelta(days=1)
        
        sales_lag_7 = working_sales_history[-7]
        sales_rolling_avg_7 = sum(working_sales_history[-7:]) / 7
        
        future_features = pd.DataFrame([{
            'store_id': target_store,
            'item_id': target_item,
            'day_of_week': loop_date.weekday(),
            'month': loop_date.month,
            'is_weekend': 1 if loop_date.weekday() in [5, 6] else 0,
            'sales_lag_7': sales_lag_7,
            'sales_rolling_avg_7': sales_rolling_avg_7,
            'price': current_price,
            'promo': current_promo
        }])
        
        future_features['store_id'] = future_features['store_id'].astype('category')
        future_features['item_id'] = future_features['item_id'].astype('category')
        
        predicted_value = int(model.predict(future_features)[0])
        working_sales_history.append(predicted_value)
        
        forecast_results.append({
            "date": loop_date.strftime("%Y-%m-%d"),
            "predicted_sales": predicted_value
        })

    # Log only the final targeted prediction to save database space
    final_prediction = forecast_results[-1]
    
    with engine.connect() as conn:
        log_query = text("""
            INSERT INTO prediction_logs (store_id, item_id, forecast_date, predicted_sales)
            VALUES (:store, :item, :f_date, :p_sales)
        """)
        conn.execute(log_query, {
            "store": target_store,
            "item": target_item,
            "f_date": final_prediction['date'],
            "p_sales": final_prediction['predicted_sales']
        })
        conn.commit()
    
    return {
        "store": target_store,
        "item": target_item,
        "forecast_date": final_prediction['date'],
        "predicted_sales": final_prediction['predicted_sales']
    }