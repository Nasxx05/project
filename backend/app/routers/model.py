"""Model training and management API endpoints."""

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models.database_models import Admission, Holiday, ModelMetric, Occupancy
from ..models.schemas import (
    LOSTrainRequest,
    ModelListResponse,
    ModelListItem,
    ModelMetricsResponse,
    OccupancyTrainRequest,
    TrainResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/model", tags=["model"])
settings = get_settings()

# Module-level model references (lazy-loaded to avoid heavy imports at startup)
_narx_model = None
_los_model = None


@router.post("/train/occupancy", response_model=TrainResponse)
async def train_occupancy_model(
    request: OccupancyTrainRequest,
    db: Session = Depends(get_db),
):
    """Train NARX bed occupancy model."""
    global _narx_model

    # Fetch occupancy data
    occ_records = (
        db.query(Occupancy)
        .filter(Occupancy.date >= request.training_start)
        .filter(Occupancy.date <= request.training_end)
        .order_by(Occupancy.date)
        .all()
    )
    if len(occ_records) < 60:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 60 days of occupancy data, found {len(occ_records)}.",
        )

    occ_df = pd.DataFrame([
        {"date": r.date, "occupancy_count": r.occupancy_count}
        for r in occ_records
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

    # Configure and train
    from ..ml.narx_forecaster import NARXOccupancyForecaster
    hyper = request.hyperparameters or {}
    config = {
        "delay": hyper.get("delay", 2),
        "hidden_layers": hyper.get("hidden_layers", 2),
        "nodes_per_layer": hyper.get("nodes_per_layer", 10),
        "epochs": hyper.get("epochs", 1000),
        "learning_rate": hyper.get("learning_rate", 0.001),
    }

    forecaster = NARXOccupancyForecaster(config=config)
    result = forecaster.train(
        occ_df,
        holidays_df,
        validation_split=request.validation_split,
    )

    # Save model
    model_path = os.path.join(settings.ml_models_path, f"narx_{forecaster.model_version}")
    forecaster.save_model(model_path)

    # Store metrics
    metric = ModelMetric(
        model_name="NARX_Occupancy",
        model_version=forecaster.model_version,
        evaluation_date=datetime.utcnow(),
        mae=result["metrics"].get("mae"),
        mape=result["metrics"].get("mape"),
        rmse=result["metrics"].get("rmse"),
        max_error=result["metrics"].get("max_error"),
        training_samples=result.get("training_samples"),
    )
    db.add(metric)
    db.commit()

    _narx_model = forecaster

    return TrainResponse(
        model_id=forecaster.model_version,
        training_history=result.get("training_history"),
        metrics=result["metrics"],
        model_path=model_path,
    )


@router.post("/train/los", response_model=TrainResponse)
async def train_los_model(
    request: LOSTrainRequest,
    db: Session = Depends(get_db),
):
    """Train length of stay predictor."""
    global _los_model

    # Fetch admissions
    admissions = (
        db.query(Admission)
        .filter(Admission.admission_date >= request.training_start)
        .filter(Admission.admission_date <= request.training_end)
        .filter(Admission.actual_los.isnot(None))
        .all()
    )
    if len(admissions) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 50 discharged patients, found {len(admissions)}.",
        )

    admissions_df = pd.DataFrame([
        {
            "patient_id": a.patient_id,
            "admission_date": a.admission_date,
            "discharge_date": a.discharge_date,
            "department": a.department,
            "age_group": a.age_group,
            "admission_type": a.admission_type,
            "primary_diagnosis_category": a.primary_diagnosis_category,
            "actual_los": a.actual_los,
        }
        for a in admissions
    ])

    # Fetch holidays
    holidays = db.query(Holiday).all()
    holidays_df = None
    if holidays:
        holidays_df = pd.DataFrame([
            {
                "date": h.date,
                "is_public_holiday": h.is_public_holiday,
                "is_school_holiday": h.is_school_holiday,
            }
            for h in holidays
        ])

    # Train
    from ..ml.los_predictor import LOSPredictor
    predictor = LOSPredictor()
    result = predictor.train(
        admissions_df,
        holidays_df,
        validation_split=request.validation_split,
        model_type=request.model_type,
    )

    # Save
    model_path = os.path.join(settings.ml_models_path, f"los_{predictor.model_version}")
    predictor.save_model(model_path)

    # Store metrics
    metric = ModelMetric(
        model_name="LOS_Predictor",
        model_version=predictor.model_version,
        evaluation_date=datetime.utcnow(),
        mae=result["metrics"].get("mae"),
        mape=result["metrics"].get("mape"),
        rmse=result["metrics"].get("rmse"),
        max_error=result["metrics"].get("max_error"),
        training_samples=result["training_info"]["training_samples"],
    )
    db.add(metric)
    db.commit()

    _los_model = predictor

    # Update the LOS router's model reference
    from . import los as los_router
    los_router._los_predictor = predictor

    return TrainResponse(
        model_id=predictor.model_version,
        metrics=result["metrics"],
        model_path=model_path,
    )


@router.get("/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    model_type: str = Query(..., regex="^(occupancy|los)$"),
    version: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Get model performance metrics."""
    model_name = "NARX_Occupancy" if model_type == "occupancy" else "LOS_Predictor"

    query = db.query(ModelMetric).filter(ModelMetric.model_name == model_name)
    if version:
        query = query.filter(ModelMetric.model_version == version)

    metric = query.order_by(ModelMetric.evaluation_date.desc()).first()
    if not metric:
        raise HTTPException(status_code=404, detail="No metrics found for this model.")

    return ModelMetricsResponse(
        mae=metric.mae,
        mape=metric.mape,
        rmse=metric.rmse,
        coverage_90=metric.coverage_90,
        interval_width=metric.avg_interval_width,
        last_updated=metric.evaluation_date.strftime("%Y-%m-%d %H:%M:%S"),
    )


@router.get("/list", response_model=ModelListResponse)
async def list_models(db: Session = Depends(get_db)):
    """List all available models with versions."""
    metrics = (
        db.query(ModelMetric)
        .order_by(ModelMetric.evaluation_date.desc())
        .all()
    )

    # Determine active model versions
    active_narx_version = _narx_model.model_version if _narx_model is not None else None
    active_los_version = _los_model.model_version if _los_model is not None else None

    models = [
        ModelListItem(
            name=m.model_name,
            version=m.model_version,
            created_at=m.evaluation_date.strftime("%Y-%m-%d %H:%M:%S"),
            metrics={"mae": m.mae, "mape": m.mape, "rmse": m.rmse},
            is_active=(
                (m.model_name == "NARX_Occupancy" and m.model_version == active_narx_version)
                or (m.model_name == "LOS_Predictor" and m.model_version == active_los_version)
            ),
        )
        for m in metrics
    ]

    return ModelListResponse(models=models)


@router.post("/activate/{model_id}")
async def activate_model(model_id: str, db: Session = Depends(get_db)):
    """Set a specific model version as active."""
    global _narx_model, _los_model

    metric = (
        db.query(ModelMetric)
        .filter(ModelMetric.model_version == model_id)
        .first()
    )
    if not metric:
        raise HTTPException(status_code=404, detail="Model version not found.")

    model_name = metric.model_name

    if model_name == "NARX_Occupancy":
        from ..ml.narx_forecaster import NARXOccupancyForecaster
        model_path = os.path.join(settings.ml_models_path, f"narx_{model_id}")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model files not found on disk.")
        _narx_model = NARXOccupancyForecaster()
        _narx_model.load_model(model_path)
        return {"message": f"Activated NARX model version {model_id}", "active_model": model_id}
    elif model_name == "LOS_Predictor":
        model_path = os.path.join(settings.ml_models_path, f"los_{model_id}")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model files not found on disk.")
        from ..ml.los_predictor import LOSPredictor
        _los_model = LOSPredictor()
        _los_model.load_model(model_path)
        from . import los as los_router
        los_router._los_predictor = _los_model
        return {"message": f"Activated LOS model version {model_id}", "active_model": model_id}
    else:
        raise HTTPException(status_code=400, detail="Unknown model type.")
