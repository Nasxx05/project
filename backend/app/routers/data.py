"""Data upload and management API endpoints."""

import io
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.database_models import Admission, Holiday, Occupancy
from ..models.schemas import (
    DataSummaryResponse,
    HistoricalOccupancyResponse,
    HolidayUploadResponse,
    UploadResponse,
)
from ..utils.converters import admissions_to_occupancy
from ..utils.validators import validate_admissions_csv, validate_holidays_csv

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["data"])


@router.post("/upload/admissions", response_model=UploadResponse)
async def upload_admissions(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload CSV of admission records."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {e}")

    validation = validate_admissions_csv(df)
    if not validation["is_valid"]:
        return UploadResponse(
            message="Validation failed",
            records_count=0,
            validation_errors=validation["errors"],
        )

    # Parse dates and calculate LOS
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])
    df["actual_los"] = (df["discharge_date"] - df["admission_date"]).dt.days

    records = []
    for _, row in df.iterrows():
        records.append(
            Admission(
                patient_id=str(row["patient_id"]),
                admission_date=row["admission_date"],
                discharge_date=row["discharge_date"],
                department=row["department"],
                age_group=row.get("age_group"),
                admission_type=row.get("admission_type"),
                primary_diagnosis_category=row.get("primary_diagnosis_category"),
                actual_los=int(row["actual_los"]),
            )
        )

    db.bulk_save_objects(records)
    db.commit()

    # Recompute occupancy from all admissions
    all_admissions = pd.read_sql(
        db.query(Admission).statement, db.bind
    )
    if not all_admissions.empty:
        occ_df = admissions_to_occupancy(
            all_admissions,
            all_admissions["admission_date"].min(),
            all_admissions["discharge_date"].max(),
        )
        # Upsert occupancy records
        db.query(Occupancy).delete()
        occ_records = [
            Occupancy(date=row["date"], occupancy_count=int(row["occupancy_count"]))
            for _, row in occ_df.iterrows()
        ]
        db.bulk_save_objects(occ_records)
        db.commit()

    logger.info("Uploaded %d admission records", len(records))

    return UploadResponse(
        message=f"Successfully uploaded {len(records)} admission records.",
        records_count=len(records),
        validation_errors=validation.get("warnings", []),
    )


@router.post("/upload/holidays", response_model=HolidayUploadResponse)
async def upload_holidays(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload CSV of holiday dates."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {e}")

    validation = validate_holidays_csv(df)
    if not validation["is_valid"]:
        raise HTTPException(status_code=400, detail=validation["errors"])

    df["date"] = pd.to_datetime(df["date"])

    records = []
    for _, row in df.iterrows():
        records.append(
            Holiday(
                date=row["date"],
                holiday_name=row["holiday_name"],
                is_public_holiday=bool(row.get("is_public_holiday", False)),
                is_school_holiday=bool(row.get("is_school_holiday", False)),
                region=row.get("region"),
            )
        )

    db.bulk_save_objects(records)
    db.commit()

    logger.info("Uploaded %d holiday records", len(records))

    return HolidayUploadResponse(
        message=f"Successfully uploaded {len(records)} holiday records.",
        records_count=len(records),
        date_range={
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
    )


@router.get("/data/summary", response_model=DataSummaryResponse)
async def get_data_summary(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Return summary statistics for all data."""
    # Admission stats
    admission_q = db.query(Admission)
    if start_date:
        admission_q = admission_q.filter(Admission.admission_date >= start_date)
    if end_date:
        admission_q = admission_q.filter(Admission.admission_date <= end_date)
    admission_count = admission_q.count()

    admission_stats = {"total_records": admission_count}
    if admission_count > 0:
        admissions = pd.read_sql(admission_q.statement, db.bind)
        admission_stats.update({
            "departments": int(admissions["department"].nunique()),
            "unique_patients": int(admissions["patient_id"].nunique()),
            "avg_los": float(admissions["actual_los"].mean()) if "actual_los" in admissions else None,
            "date_range": {
                "start": str(admissions["admission_date"].min()),
                "end": str(admissions["admission_date"].max()),
            },
        })

    # Occupancy stats
    occ_q = db.query(Occupancy)
    if start_date:
        occ_q = occ_q.filter(Occupancy.date >= start_date)
    if end_date:
        occ_q = occ_q.filter(Occupancy.date <= end_date)
    occ_count = occ_q.count()

    occupancy_stats = {"total_records": occ_count}
    if occ_count > 0:
        occ_df = pd.read_sql(occ_q.statement, db.bind)
        occupancy_stats.update({
            "mean_occupancy": float(occ_df["occupancy_count"].mean()),
            "max_occupancy": int(occ_df["occupancy_count"].max()),
            "min_occupancy": int(occ_df["occupancy_count"].min()),
        })

    # Holiday stats
    holiday_count = db.query(Holiday).count()

    return DataSummaryResponse(
        occupancy_stats=occupancy_stats,
        admission_stats=admission_stats,
        data_completeness={
            "admissions": admission_count > 0,
            "occupancy": occ_count > 0,
            "holidays": holiday_count > 0,
        },
    )


@router.get("/occupancy/historical", response_model=HistoricalOccupancyResponse)
async def get_historical_occupancy(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    granularity: str = Query("day", regex="^(day|week|month)$"),
    db: Session = Depends(get_db),
):
    """Return historical occupancy data."""
    query = db.query(Occupancy).order_by(Occupancy.date)
    if start_date:
        query = query.filter(Occupancy.date >= start_date)
    if end_date:
        query = query.filter(Occupancy.date <= end_date)

    records = query.all()
    if not records:
        return HistoricalOccupancyResponse(dates=[], occupancy=[], statistics={})

    df = pd.DataFrame([{"date": r.date, "occupancy_count": r.occupancy_count} for r in records])
    df["date"] = pd.to_datetime(df["date"])

    if granularity == "week":
        df = df.set_index("date").resample("W").mean().reset_index()
        df["occupancy_count"] = df["occupancy_count"].round().astype(int)
    elif granularity == "month":
        df = df.set_index("date").resample("M").mean().reset_index()
        df["occupancy_count"] = df["occupancy_count"].round().astype(int)

    stats = {
        "mean": float(df["occupancy_count"].mean()),
        "std": float(df["occupancy_count"].std()),
        "min": int(df["occupancy_count"].min()),
        "max": int(df["occupancy_count"].max()),
    }

    return HistoricalOccupancyResponse(
        dates=[d.strftime("%Y-%m-%d") for d in df["date"]],
        occupancy=df["occupancy_count"].tolist(),
        statistics=stats,
    )


@router.delete("/data/clear/{data_type}")
async def clear_data(
    data_type: str,
    confirm: str = Query(..., description="Pass 'yes' to confirm deletion"),
    db: Session = Depends(get_db),
):
    """Clear specific data type."""
    if confirm != "yes":
        raise HTTPException(status_code=400, detail="Pass confirm=yes to proceed.")

    if data_type == "admissions":
        count = db.query(Admission).delete()
        db.query(Occupancy).delete()
        db.commit()
        return {"message": f"Deleted {count} admission records and associated occupancy data."}
    elif data_type == "holidays":
        count = db.query(Holiday).delete()
        db.commit()
        return {"message": f"Deleted {count} holiday records."}
    else:
        raise HTTPException(status_code=400, detail="data_type must be 'admissions' or 'holidays'.")
