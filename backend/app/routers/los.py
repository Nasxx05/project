"""Length of Stay prediction API endpoints."""

import io
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.database_models import Admission, Holiday, LOSPrediction
from ..models.schemas import LOSPredictRequest, LOSPredictResponse, LOSStatisticsResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/los", tags=["los"])

# Module-level model instance (lazy-loaded)
_los_predictor = None


def _get_predictor():
    from ..ml.los_predictor import LOSPredictor
    global _los_predictor
    if _los_predictor is None:
        _los_predictor = LOSPredictor()
    return _los_predictor


@router.post("/predict", response_model=LOSPredictResponse)
async def predict_los(
    request: LOSPredictRequest,
    db: Session = Depends(get_db),
):
    """Predict length of stay for a patient."""
    predictor = _get_predictor()
    if predictor.model is None:
        raise HTTPException(
            status_code=400,
            detail="LOS model not trained. Train the model first via /api/v1/model/train/los.",
        )

    patient_data = {
        "age_group": request.age_group,
        "admission_type": request.admission_type,
        "department": request.department,
        "primary_diagnosis_category": request.diagnosis_category,
        "admission_date": pd.Timestamp.now(),
        "is_holiday_period": int(request.is_holiday_period),
    }

    if request.admission_day_of_week is not None:
        patient_data["admission_day_of_week"] = request.admission_day_of_week
    if request.admission_hour is not None:
        patient_data["admission_hour"] = request.admission_hour

    result = predictor.predict_with_intervals(patient_data, confidence_level=0.90)

    # Count similar cases
    similar_q = db.query(Admission).filter(Admission.department == request.department)
    if request.diagnosis_category:
        similar_q = similar_q.filter(
            Admission.primary_diagnosis_category == request.diagnosis_category
        )
    similar_count = similar_q.count()

    # Store prediction
    los_pred = LOSPrediction(
        patient_id=request.patient_id,
        admission_date=pd.Timestamp.now(),
        predicted_los=result["predicted_los"],
        lower_bound=result["lower_bound"],
        upper_bound=result["upper_bound"],
        features_used=patient_data,
        model_version=predictor.model_version,
    )
    db.add(los_pred)
    db.commit()

    return LOSPredictResponse(
        predicted_los=result["predicted_los"],
        lower_bound=result["lower_bound"],
        upper_bound=result["upper_bound"],
        confidence_level=result["confidence_level"],
        similar_cases=similar_count,
    )


@router.post("/batch-predict")
async def batch_predict_los(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload CSV of patient characteristics and return LOS predictions."""
    predictor = _get_predictor()
    if predictor.model is None:
        raise HTTPException(
            status_code=400,
            detail="LOS model not trained. Train the model first.",
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {e}")

    # Ensure minimum required columns
    required = {"department"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing)}",
        )

    # Fill defaults
    if "admission_date" not in df.columns:
        df["admission_date"] = pd.Timestamp.now()

    result = predictor.predict_with_intervals(df, confidence_level=0.90)

    predictions = result.get("predictions", [])
    summary = {}
    if predictions:
        pred_values = [p["predicted_los"] for p in predictions]
        summary = {
            "count": len(predictions),
            "mean_los": round(sum(pred_values) / len(pred_values), 1),
            "min_los": min(pred_values),
            "max_los": max(pred_values),
        }

    return {"predictions": predictions, "summary_statistics": summary}


@router.get("/statistics", response_model=LOSStatisticsResponse)
async def get_los_statistics(
    department: Optional[str] = Query(None),
    diagnosis_category: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Return LOS statistics from historical admissions."""
    query = db.query(Admission).filter(Admission.actual_los.isnot(None))

    if department:
        query = query.filter(Admission.department == department)
    if diagnosis_category:
        query = query.filter(Admission.primary_diagnosis_category == diagnosis_category)
    if start_date:
        query = query.filter(Admission.admission_date >= start_date)
    if end_date:
        query = query.filter(Admission.admission_date <= end_date)

    records = query.all()
    if not records:
        raise HTTPException(status_code=404, detail="No matching admission records found.")

    los_values = pd.Series([r.actual_los for r in records if r.actual_los is not None])

    return LOSStatisticsResponse(
        mean_los=round(float(los_values.mean()), 1),
        median_los=round(float(los_values.median()), 1),
        p25=round(float(los_values.quantile(0.25)), 1),
        p75=round(float(los_values.quantile(0.75)), 1),
        p90=round(float(los_values.quantile(0.90)), 1),
        distribution={
            "min": int(los_values.min()),
            "max": int(los_values.max()),
            "std": round(float(los_values.std()), 1),
            "histogram": los_values.value_counts().sort_index().head(30).to_dict(),
        },
    )
