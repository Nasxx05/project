# Hospital Bed Occupancy and Length of Stay Forecasting System

A web-based decision support system for hospital resource planning that predicts daily bed occupancy and individual patient length of stay with uncertainty quantification through prediction intervals.

## Features

- **Bed Occupancy Forecasting**: NARX neural network predicts daily occupancy for up to 365 days with 80%/90%/95% confidence intervals using bootstrap ensemble
- **Length of Stay Prediction**: Random Forest / Gradient Boosting models predict individual patient LOS with prediction intervals
- **Data Upload**: CSV upload for admission records and holiday calendars with validation
- **Analytics Dashboard**: Seasonal patterns, forecast accuracy, capacity planning, and feature importance
- **Model Management**: Train, evaluate, compare, and activate different model versions

## Tech Stack

- **Backend**: Python FastAPI, SQLAlchemy, PostgreSQL
- **ML**: TensorFlow/Keras (NARX), scikit-learn (LOS)
- **Frontend**: React 18, Material-UI 5, Chart.js
- **Deployment**: Docker + docker-compose

## Quick Start

```bash
# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Generate Sample Data

```bash
cd scripts
pip install pandas numpy
python generate_sample_data.py --records 5000 --start 2020-01-01 --end 2023-12-31
```

Then upload the generated CSVs through the web UI.

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── config.py            # Application settings
│   │   ├── database.py          # SQLAlchemy engine and session
│   │   ├── models/
│   │   │   ├── database_models.py  # ORM models (6 tables)
│   │   │   └── schemas.py         # Pydantic request/response schemas
│   │   ├── routers/
│   │   │   ├── data.py           # Upload and data management endpoints
│   │   │   ├── forecast.py       # Bed occupancy forecast endpoints
│   │   │   ├── los.py            # Length of stay prediction endpoints
│   │   │   ├── model.py          # Model training and management
│   │   │   └── analytics.py      # Analytics and reporting
│   │   ├── ml/
│   │   │   ├── narx_forecaster.py  # NARX occupancy model
│   │   │   └── los_predictor.py    # LOS prediction model
│   │   └── utils/
│   │       ├── validators.py       # CSV validation
│   │       ├── converters.py       # Admissions to occupancy
│   │       ├── feature_engineering.py  # Temporal and holiday features
│   │       └── data_processing.py     # Metrics, outliers, missing data
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main app with routing and navigation
│   │   ├── components/
│   │   │   ├── Dashboard.js     # Forecast chart and controls
│   │   │   ├── DataUpload.js    # CSV upload with drag-and-drop
│   │   │   ├── LOSPrediction.js # Patient LOS prediction form
│   │   │   ├── ModelTraining.js # Model training interface
│   │   │   └── Analytics.js     # Charts and analytics views
│   │   └── services/
│   │       └── api.js           # Axios API client
│   ├── Dockerfile
│   └── nginx.conf
├── scripts/
│   └── generate_sample_data.py  # Synthetic data generator
├── docker-compose.yml
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/upload/admissions | Upload admission CSV |
| POST | /api/v1/upload/holidays | Upload holiday CSV |
| GET | /api/v1/data/summary | Data summary statistics |
| GET | /api/v1/occupancy/historical | Historical occupancy |
| POST | /api/v1/forecast/bed/generate | Generate bed forecast |
| GET | /api/v1/forecast/bed/latest | Latest forecast |
| POST | /api/v1/los/predict | Predict patient LOS |
| POST | /api/v1/los/batch-predict | Batch LOS prediction |
| GET | /api/v1/los/statistics | LOS statistics |
| POST | /api/v1/model/train/occupancy | Train NARX model |
| POST | /api/v1/model/train/los | Train LOS model |
| GET | /api/v1/analytics/seasonal-patterns | Seasonal analysis |
| GET | /api/v1/analytics/capacity-planning | Capacity recommendations |

Full interactive API documentation available at `/docs` when the backend is running.

## References

Based on the methodology from:
- Kutafina, E. et al. (2019). "Forecast of bed occupancy using a NARX model with prediction intervals."
