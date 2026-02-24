"""SQLAlchemy ORM models for the hospital forecasting system."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, JSON, Index
)
from ..database import Base


class Admission(Base):
    __tablename__ = "admissions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True, nullable=False)
    admission_date = Column(DateTime, nullable=False)
    discharge_date = Column(DateTime, nullable=False)
    department = Column(String, nullable=False)
    age_group = Column(String, nullable=True)
    admission_type = Column(String, nullable=True)
    primary_diagnosis_category = Column(String, nullable=True)
    actual_los = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_admissions_dates", "admission_date", "discharge_date"),
    )


class Occupancy(Base):
    __tablename__ = "occupancy"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, unique=True, nullable=False, index=True)
    occupancy_count = Column(Integer, nullable=False)
    available_beds = Column(Integer, nullable=True)
    occupancy_rate = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Holiday(Base):
    __tablename__ = "holidays"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, unique=True, nullable=False, index=True)
    holiday_name = Column(String, nullable=False)
    is_public_holiday = Column(Boolean, default=False)
    is_school_holiday = Column(Boolean, default=False)
    region = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BedPrediction(Base):
    __tablename__ = "bed_predictions"

    id = Column(Integer, primary_key=True, index=True)
    forecast_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False, index=True)
    predicted_occupancy = Column(Float, nullable=False)
    lower_bound_90 = Column(Float, nullable=True)
    upper_bound_90 = Column(Float, nullable=True)
    lower_bound_80 = Column(Float, nullable=True)
    upper_bound_80 = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    mae = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_bed_predictions_forecast", "forecast_date", "target_date"),
    )


class LOSPrediction(Base):
    __tablename__ = "los_predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True, nullable=True)
    admission_date = Column(DateTime, nullable=True)
    predicted_los = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    actual_los = Column(Integer, nullable=True)
    prediction_error = Column(Float, nullable=True)
    features_used = Column(JSON, nullable=True)
    model_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    evaluation_date = Column(DateTime, nullable=False)
    mae = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    max_error = Column(Float, nullable=True)
    coverage_90 = Column(Float, nullable=True)
    avg_interval_width = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_model_metrics_name_version", "model_name", "model_version"),
    )
