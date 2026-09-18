# Enterprise Demand Forecasting — Full-Stack ML Pipeline

An end-to-end, production-ready Full-Stack Machine Learning and MLOps system for retail demand forecasting. Engineered to transition from localized single-item models to a scalable multi-tenant global forecasting architecture backed by cloud PostgreSQL, FastAPI, recursive forecasting loops, and automated drift monitoring.

---

## System Architecture & Evolution
1. **Data Layer**: Neon PostgreSQL cloud database housing 4.5M+ retail transaction records.
2. **Feature Engineering**: Grouped sequential time-series features (`sales_lag_7`, `sales_rolling_avg_7`) preventing data leakage across multi-product boundaries.
3. **Modeling Strategy**:
   - **Local Model**: Specialized single-item baseline (`MAPE: 6.70%`)
   - **Global Model**: Universal multi-tenant model handling 10 stores & 50 items simultaneously (`MAPE: 12.25%`), balancing the Bias-Variance tradeoff for enterprise scalability.
4. **Backend API**: FastAPI server with CORS middleware, Pydantic validation, recursive multi-step forecasting engine, and automated audit logging.
5. **Frontend Dashboard**: Responsive single-page UI built with HTML5 & Tailwind CSS featuring calendar-driven target date inference.
6. **MLOps & Drift Monitoring**: Audit trail logging (`prediction_logs`), actuals-to-prediction join evaluation, and `subprocess`-driven retraining triggers (`MAPE > 15%`).

---

## Key Performance Metrics
| Model Type | Scope | Evaluation Metric (MAPE) | Business Tradeoff |
| :--- | :--- | :--- | :--- |
| **Local Model** | 1 Store / 1 Item | **6.70%** | High hyper-specific accuracy; unscalable management overhead. |
| **Global Model** | 10 Stores / 50 Items | **12.25%** | Universal generalizability; ideal for enterprise retail operations. |

---

## Project Directory Structure
```text
ML FULL STACK/
├── 3_model_training.py          # Local model training script (6.70% MAPE)
├── 3_model_training_global.py   # Global multi-tenant model training script (12.25% MAPE)
├── 4_inference.py               # Local single-step inference script
├── 4_inference_global.py        # Global categorical inference script
├── 5_api.py                     # FastAPI production server with recursive forecasting & audit logging
├── 6_drift_monitor.py           # MLOps drift evaluation & automated retraining watchdog
├── index.html                   # Tailwind CSS frontend interactive dashboard
├── requirements.txt             # Python dependencies
└── .gitignore                   # Credential & artifact isolation
