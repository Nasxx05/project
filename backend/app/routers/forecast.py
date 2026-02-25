"""Bed occupancy forecasting API endpoints."""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models.database_models import BedPrediction, Holiday, ModelMetric, Occupancy
from ..models.schemas import (
    ForecastConfig,
    ForecastHistoryResponse,
    ForecastRequest,
    ForecastResponse,
    PredictionPoint,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/forecast/bed", tags=["forecast"])
settings = get_settings()

# Module-level model instance (loaded on first use)
_forecaster = None


def _get_forecaster():
    from ..ml.narx_forecaster import NARXOccupancyForecaster
    global _forecaster
    if _forecaster is None:
        _forecaster = NARXOccupancyForecaster()
    return _forecaster


@router.post("/generate", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    db: Session = Depends(get_db),
):
    """Generate bed occupancy forecast with prediction intervals."""
    # Fetch occupancy data
    occupancy_records = (
        db.query(Occupancy).order_by(Occupancy.date).all()
    )
    if not occupancy_records or len(occupancy_records) < 30:
        raise HTTPException(
            status_code=400,
            detail="Insufficient occupancy data. Upload admissions first (need at least 30 days).",
        )

    occ_df = pd.DataFrame([
        {"date": r.date, "occupancy_count": r.occupancy_count}
        for r in occupancy_records
    ])

    # Fetch holidays
    holidays = db.query(Holiday).all()
    holidays_df = None
    if holidays:
        holidays_df = pd.DataFrame([
            {
                "date": h.date,
                "holiday_name": h.holiday_name,
                "is_public_holiday": h.is_public_holiday,
                "is_school_holiday": h.is_school_holiday,
            }
            for h in holidays
        ])

    # Configure model
    from ..ml.narx_forecaster import NARXOccupancyForecaster
    config = request.model_config or ForecastConfig()
    forecaster = NARXOccupancyForecaster(config={
        "delay": config.delay,
        "history_days": config.history_days,
        "ensemble_size": config.ensemble_size,
    })

    # Train on available data
    logger.info("Training NARX model for forecast...")
    train_result = forecaster.train(occ_df, holidays_df)

    # Generate forecast
    logger.info("Generating %d-day forecast...", request.forecast_days)
    forecast_result = forecaster.predict_with_intervals(
        occ_df,
        holidays_df,
        forecast_days=request.forecast_days,
        confidence_levels=request.confidence_levels,
        n_bootstrap=config.ensemble_size,
    )

    # Store predictions
    forecast_date = datetime.utcnow()
    predictions = []
    for i, date_str in enumerate(forecast_result["dates"]):
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        pred = BedPrediction(
            forecast_date=forecast_date,
            target_date=target_date,
            predicted_occupancy=forecast_result["predictions"][i],
            lower_bound_90=forecast_result["intervals"].get("90", {}).get("lower", [None] * len(forecast_result["dates"]))[i],
            upper_bound_90=forecast_result["intervals"].get("90", {}).get("upper", [None] * len(forecast_result["dates"]))[i],
            lower_bound_80=forecast_result["intervals"].get("80", {}).get("lower", [None] * len(forecast_result["dates"]))[i],
            upper_bound_80=forecast_result["intervals"].get("80", {}).get("upper", [None] * len(forecast_result["dates"]))[i],
            model_version=forecaster.model_version,
        )
        db.add(pred)
        predictions.append(PredictionPoint(
            date=date_str,
            predicted=round(forecast_result["predictions"][i], 1),
            lower_90=round(forecast_result["intervals"].get("90", {}).get("lower", [0] * len(forecast_result["dates"]))[i], 1),
            upper_90=round(forecast_result["intervals"].get("90", {}).get("upper", [0] * len(forecast_result["dates"]))[i], 1),
            lower_80=round(forecast_result["intervals"].get("80", {}).get("lower", [0] * len(forecast_result["dates"]))[i], 1),
            upper_80=round(forecast_result["intervals"].get("80", {}).get("upper", [0] * len(forecast_result["dates"]))[i], 1),
        ))

    # Store model metrics
    metric = ModelMetric(
        model_name="NARX_Occupancy",
        model_version=forecaster.model_version,
        evaluation_date=forecast_date,
        mae=train_result["metrics"].get("mae"),
        mape=train_result["metrics"].get("mape"),
        rmse=train_result["metrics"].get("rmse"),
        max_error=train_result["metrics"].get("max_error"),
        training_samples=train_result.get("training_samples"),
    )
    db.add(metric)
    db.commit()

    # Get the forecast ID (from the first prediction)
    db.refresh(predictions[0] if predictions else pred)
    forecast_id = db.query(func.max(BedPrediction.id)).scalar() or 0

    # Update global model reference
    global _forecaster
    _forecaster = forecaster

    return ForecastResponse(
        forecast_id=forecast_id,
        predictions=predictions,
        model_metrics=train_result["metrics"],
        computation_time=forecast_result.get("computation_time"),
    )


@router.get("/latest")
async def get_latest_forecast(
    limit: int = Query(60, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Get most recent bed occupancy forecast."""
    latest_date = db.query(func.max(BedPrediction.forecast_date)).scalar()
    if not latest_date:
        raise HTTPException(status_code=404, detail="No forecasts available.")

    records = (
        db.query(BedPrediction)
        .filter(BedPrediction.forecast_date == latest_date)
        .order_by(BedPrediction.target_date)
        .limit(limit)
        .all()
    )

    predictions = [
        {
            "date": r.target_date.strftime("%Y-%m-%d"),
            "predicted": r.predicted_occupancy,
            "lower_90": r.lower_bound_90,
            "upper_90": r.upper_bound_90,
            "lower_80": r.lower_bound_80,
            "upper_80": r.upper_bound_80,
        }
        for r in records
    ]

    # Get associated metrics
    metric = (
        db.query(ModelMetric)
        .filter(ModelMetric.model_name == "NARX_Occupancy")
        .order_by(ModelMetric.evaluation_date.desc())
        .first()
    )

    return {
        "forecast_date": latest_date.strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": predictions,
        "model_version": records[0].model_version if records else None,
        "metrics": {
            "mae": metric.mae,
            "mape": metric.mape,
            "rmse": metric.rmse,
        } if metric else None,
    }


@router.get("/history", response_model=ForecastHistoryResponse)
async def get_forecast_history(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get all historical forecast summaries."""
    # Get distinct forecast dates
    query = db.query(
        BedPrediction.forecast_date,
        BedPrediction.model_version,
        func.count(BedPrediction.id).label("prediction_count"),
        func.avg(BedPrediction.predicted_occupancy).label("avg_prediction"),
    ).group_by(BedPrediction.forecast_date, BedPrediction.model_version)

    if start_date:
        query = query.filter(BedPrediction.forecast_date >= start_date)
    if end_date:
        query = query.filter(BedPrediction.forecast_date <= end_date)

    total = query.count()
    results = (
        query.order_by(BedPrediction.forecast_date.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    forecasts = [
        {
            "forecast_date": r.forecast_date.strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": r.model_version,
            "prediction_count": r.prediction_count,
            "avg_prediction": round(float(r.avg_prediction), 1),
        }
        for r in results
    ]

    return ForecastHistoryResponse(
        forecasts=forecasts,
        total_count=total,
        page_info={"page": page, "limit": limit, "total_pages": (total + limit - 1) // limit},
    )


@router.get("/{forecast_id}")
async def get_forecast_by_id(
    forecast_id: int,
    db: Session = Depends(get_db),
):
    """Get specific forecast and compare with actuals if available."""
    prediction = db.query(BedPrediction).filter(BedPrediction.id == forecast_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Forecast not found.")

    # Get all predictions from the same forecast run
    records = (
        db.query(BedPrediction)
        .filter(BedPrediction.forecast_date == prediction.forecast_date)
        .order_by(BedPrediction.target_date)
        .all()
    )

    # Check for actual occupancy values
    actuals = []
    errors = []
    for r in records:
        actual = (
            db.query(Occupancy)
            .filter(func.date(Occupancy.date) == r.target_date.date())
            .first()
        )
        if actual:
            actuals.append({
                "date": r.target_date.strftime("%Y-%m-%d"),
                "actual": actual.occupancy_count,
            })
            errors.append({
                "date": r.target_date.strftime("%Y-%m-%d"),
                "error": abs(actual.occupancy_count - r.predicted_occupancy),
            })

    forecast = {
        "forecast_date": prediction.forecast_date.strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": prediction.model_version,
        "predictions": [
            {
                "date": r.target_date.strftime("%Y-%m-%d"),
                "predicted": r.predicted_occupancy,
                "lower_90": r.lower_bound_90,
                "upper_90": r.upper_bound_90,
                "lower_80": r.lower_bound_80,
                "upper_80": r.upper_bound_80,
            }
            for r in records
        ],
    }

    return {"forecast": forecast, "actuals": actuals, "errors": errors}
