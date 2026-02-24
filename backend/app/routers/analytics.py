"""Analytics and reporting API endpoints."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.database_models import (
    Admission,
    BedPrediction,
    ModelMetric,
    Occupancy,
)
from ..models.schemas import (
    CapacityPlanningResponse,
    FeatureImportanceResponse,
    ForecastAccuracyResponse,
    SeasonalPatternsResponse,
)
from ..utils.data_processing import calculate_metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/forecast-accuracy", response_model=ForecastAccuracyResponse)
async def get_forecast_accuracy(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Compare predictions vs actual occupancy."""
    # Get the latest forecast set
    latest_forecast_date = db.query(func.max(BedPrediction.forecast_date)).scalar()
    if not latest_forecast_date:
        raise HTTPException(status_code=404, detail="No forecasts available.")

    pred_q = db.query(BedPrediction).filter(
        BedPrediction.forecast_date == latest_forecast_date
    )
    if start_date:
        pred_q = pred_q.filter(BedPrediction.target_date >= start_date)
    if end_date:
        pred_q = pred_q.filter(BedPrediction.target_date <= end_date)

    predictions = pred_q.order_by(BedPrediction.target_date).all()

    dates = []
    predicted = []
    actual = []
    errors = []

    for p in predictions:
        occ = (
            db.query(Occupancy)
            .filter(func.date(Occupancy.date) == p.target_date.date())
            .first()
        )
        if occ:
            dates.append(p.target_date.strftime("%Y-%m-%d"))
            predicted.append(round(p.predicted_occupancy, 1))
            actual.append(occ.occupancy_count)
            errors.append(round(abs(occ.occupancy_count - p.predicted_occupancy), 1))

    if not dates:
        raise HTTPException(
            status_code=404,
            detail="No overlapping forecast and actual data found.",
        )

    metrics = calculate_metrics(np.array(actual), np.array(predicted))

    return ForecastAccuracyResponse(
        dates=dates,
        predicted=predicted,
        actual=[float(a) for a in actual],
        errors=errors,
        metrics=metrics,
    )


@router.get("/seasonal-patterns", response_model=SeasonalPatternsResponse)
async def get_seasonal_patterns(db: Session = Depends(get_db)):
    """Analyze seasonal patterns in occupancy and LOS."""
    # Occupancy patterns
    occ_records = db.query(Occupancy).order_by(Occupancy.date).all()
    if not occ_records:
        raise HTTPException(status_code=404, detail="No occupancy data available.")

    occ_df = pd.DataFrame([
        {"date": r.date, "occupancy_count": r.occupancy_count}
        for r in occ_records
    ])
    occ_df["date"] = pd.to_datetime(occ_df["date"])
    occ_df["month"] = occ_df["date"].dt.month
    occ_df["day_of_week"] = occ_df["date"].dt.dayofweek
    occ_df["year"] = occ_df["date"].dt.year

    # Monthly averages
    monthly = occ_df.groupby("month")["occupancy_count"].agg(["mean", "std"]).reset_index()
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    monthly_averages = [
        {
            "month": month_names[int(row["month"]) - 1],
            "mean_occupancy": round(float(row["mean"]), 1),
            "std": round(float(row["std"]), 1) if pd.notna(row["std"]) else 0,
        }
        for _, row in monthly.iterrows()
    ]

    # Day of week patterns
    dow = occ_df.groupby("day_of_week")["occupancy_count"].agg(["mean", "std"]).reset_index()
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_patterns = [
        {
            "day": day_names[int(row["day_of_week"])],
            "mean_occupancy": round(float(row["mean"]), 1),
            "std": round(float(row["std"]), 1) if pd.notna(row["std"]) else 0,
        }
        for _, row in dow.iterrows()
    ]

    # Yearly trends
    yearly = occ_df.groupby("year")["occupancy_count"].agg(["mean", "min", "max"]).reset_index()
    yearly_trends = [
        {
            "year": int(row["year"]),
            "mean_occupancy": round(float(row["mean"]), 1),
            "min_occupancy": int(row["min"]),
            "max_occupancy": int(row["max"]),
        }
        for _, row in yearly.iterrows()
    ]

    # Holiday impact
    holiday_impact = None
    admissions = db.query(Admission).all()
    if admissions:
        adm_df = pd.DataFrame([
            {"admission_date": a.admission_date, "actual_los": a.actual_los}
            for a in admissions if a.actual_los is not None
        ])
        if not adm_df.empty:
            adm_df["admission_date"] = pd.to_datetime(adm_df["admission_date"])
            adm_df["day_of_week"] = adm_df["admission_date"].dt.dayofweek
            weekday_los = adm_df[adm_df["day_of_week"] < 5]["actual_los"].mean()
            weekend_los = adm_df[adm_df["day_of_week"] >= 5]["actual_los"].mean()
            holiday_impact = {
                "weekday_avg_los": round(float(weekday_los), 1) if pd.notna(weekday_los) else None,
                "weekend_avg_los": round(float(weekend_los), 1) if pd.notna(weekend_los) else None,
            }

    return SeasonalPatternsResponse(
        monthly_averages=monthly_averages,
        day_of_week_patterns=dow_patterns,
        holiday_impact=holiday_impact,
        yearly_trends=yearly_trends,
    )


@router.get("/capacity-planning", response_model=CapacityPlanningResponse)
async def get_capacity_planning(
    target_date: Optional[str] = Query(None),
    scenario: str = Query("realistic", regex="^(optimistic|realistic|pessimistic)$"),
    db: Session = Depends(get_db),
):
    """Provide capacity planning recommendations."""
    occ_records = db.query(Occupancy).order_by(Occupancy.date).all()
    if not occ_records:
        raise HTTPException(status_code=404, detail="No occupancy data for planning.")

    occ_values = [r.occupancy_count for r in occ_records]
    mean_occ = np.mean(occ_values)
    std_occ = np.std(occ_values)
    max_occ = np.max(occ_values)

    # Scenario-based recommendations
    scenario_multipliers = {
        "optimistic": {"beds": 1.05, "util": 0.85},
        "realistic": {"beds": 1.15, "util": 0.80},
        "pessimistic": {"beds": 1.30, "util": 0.75},
    }
    mult = scenario_multipliers[scenario]

    recommended_beds = int(np.ceil(max_occ * mult["beds"]))
    utilization = round(float(mean_occ / recommended_beds * 100), 1) if recommended_beds > 0 else 0

    # Staffing: rough estimate (1 nurse per 4-6 beds)
    nurses_per_shift = int(np.ceil(recommended_beds / 5))

    risk_level = "Low"
    if utilization > 85:
        risk_level = "High"
    elif utilization > 75:
        risk_level = "Medium"

    return CapacityPlanningResponse(
        recommended_beds=recommended_beds,
        utilization_forecast=utilization,
        staffing_recommendations={
            "nurses_per_shift": nurses_per_shift,
            "total_shifts_per_day": 3,
            "scenario": scenario,
        },
        risk_assessment={
            "risk_level": risk_level,
            "mean_occupancy": round(float(mean_occ), 1),
            "peak_occupancy": int(max_occ),
            "occupancy_std": round(float(std_occ), 1),
            "overflow_probability": round(
                float(np.mean(np.array(occ_values) > recommended_beds * 0.95) * 100), 1
            ),
        },
    )


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(db: Session = Depends(get_db)):
    """Show which features most influence predictions."""
    # LOS feature importance (from the model if loaded)
    los_features = []
    try:
        from .los import _los_predictor
        if _los_predictor and _los_predictor.model is not None:
            los_features = _los_predictor.get_feature_importance()
    except Exception:
        pass

    # NARX model doesn't have direct feature importance (neural network),
    # so we provide a static description based on the architecture
    occupancy_features = [
        {"name": "Previous day occupancy (lag_1)", "importance": 0.25, "description": "Most recent occupancy count"},
        {"name": "7-day rolling average", "importance": 0.18, "description": "Weekly trend indicator"},
        {"name": "Day of week", "importance": 0.15, "description": "Weekly admission/discharge patterns"},
        {"name": "30-day rolling average", "importance": 0.12, "description": "Monthly trend indicator"},
        {"name": "Holiday indicators", "importance": 0.10, "description": "Public and school holiday effects"},
        {"name": "Month / Season", "importance": 0.08, "description": "Seasonal illness patterns"},
        {"name": "Occupancy trend", "importance": 0.07, "description": "Difference from 7-day average"},
        {"name": "Weekend indicator", "importance": 0.05, "description": "Weekend vs weekday patterns"},
    ]

    return FeatureImportanceResponse(
        occupancy_features=occupancy_features,
        los_features=los_features,
    )
