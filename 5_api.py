from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from datetime import timedelta, datetime

app = FastAPI(
    title="Enterprise Demand Forecasting API",
    description="Machine Learning Demand Forecasting System powered by XGBoost"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

DEFAULT_DB_URL = "postgresql://neondb_owner:npg_DRic57rwNIgE@ep-billowing-river-b32qv1z9.c-4.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
raw_db_url = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URL_POOLED") or DEFAULT_DB_URL
db_url = raw_db_url.replace("postgres://", "postgresql://", 1)

try:
    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=300)
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
except Exception as e:
    print(f"Database setup warning: {e}")
    engine = None

model_path = 'xgboost_global_demand_model.pkl'
print("Loading XGBoost model into memory...")
try:
    model = joblib.load(model_path)
    print("[OK] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    model = None

class ForecastRequest(BaseModel):
    store_id: str
    item_id: str
    target_date: str

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Enterprise Demand Forecasting API is active."}

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "database_connected": engine is not None
    }

@app.post("/predict")
def predict_demand(request: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="ML Model is not loaded on server.")
        
    target_store = request.store_id.strip()
    target_item = request.item_id.strip()
    
    # Parse target date
    try:
        requested_date = datetime.strptime(request.target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if engine is None:
        raise HTTPException(status_code=500, detail="Database connection unavailable.")
        
    query = f"""
        SELECT date, sales, price, promo
        FROM sales_data
        WHERE store_id = '{target_store}' AND item_id = '{target_item}'
        ORDER BY date DESC
        LIMIT 7
    """
    try:
        df_recent = pd.read_sql_query(query, engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

    if df_recent.empty:
        raise HTTPException(status_code=404, detail=f"No sales history found for {target_store} / {target_item}.")
        
    df_recent['date'] = pd.to_datetime(df_recent['date'])
    df_recent = df_recent.sort_values('date', ascending=True).reset_index(drop=True)
    
    current_date = df_recent['date'].iloc[-1].date()
    horizon = (requested_date - current_date).days
    
    if horizon <= 0:
        raise HTTPException(
            status_code=400, 
            detail=f"Target date must be strictly in the future of historical baseline ({current_date.strftime('%Y-%m-%d')})."
        )
        
    if horizon > 3650:
        raise HTTPException(status_code=400, detail="Target date is too far in the future (max 10 years).")

    working_sales_history = df_recent['sales'].tolist()
    current_price = float(df_recent['price'].iloc[-1])
    current_promo = int(df_recent['promo'].iloc[-1])
    
    forecast_results = []
    loop_date = current_date
    
    for i in range(horizon):
        loop_date = loop_date + timedelta(days=1)
        
        sales_lag_7 = working_sales_history[-7]
        sales_rolling_avg_7 = sum(working_sales_history[-7:]) / 7.0
        
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
        
        predicted_value = max(0, int(model.predict(future_features)[0]))
        working_sales_history.append(predicted_value)
        
        forecast_results.append({
            "date": loop_date.strftime("%Y-%m-%d"),
            "predicted_sales": predicted_value
        })

    final_prediction = forecast_results[-1]
    
    # Log prediction safely
    try:
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
    except Exception as e:
        print(f"Warning: Failed to insert prediction log: {e}")

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    target_dt = datetime.strptime(final_prediction['date'], "%Y-%m-%d")

    return {
        "store": target_store,
        "item": target_item,
        "forecast_date": final_prediction['date'],
        "predicted_sales": final_prediction['predicted_sales'],
        "days_ahead": horizon,
        "baseline_date": current_date.strftime("%Y-%m-%d"),
        "day_of_week": day_names[target_dt.weekday()],
        "is_weekend": target_dt.weekday() in [5, 6],
        "rolling_avg_7": round(sum(working_sales_history[-7:]) / 7.0, 1)
    }