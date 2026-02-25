"""Pydantic schemas for request/response validation."""

from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel, Field


# --- Data Upload ---

class UploadResponse(BaseModel):
    message: str
    records_count: int
    validation_errors: list[str] = []


class HolidayUploadResponse(BaseModel):
    message: str
    records_count: int
    date_range: Optional[dict] = None


# --- Data Summary ---

class DataSummaryResponse(BaseModel):
    occupancy_stats: Optional[dict] = None
    admission_stats: Optional[dict] = None
    data_completeness: Optional[dict] = None


class HistoricalOccupancyResponse(BaseModel):
    dates: list[str]
    occupancy: list[int]
    statistics: Optional[dict] = None


# --- Bed Occupancy Forecast ---

class ForecastConfig(BaseModel):
    delay: int = Field(default=2, ge=1, le=7)
    history_days: int = Field(default=365, ge=180, le=1825)
    ensemble_size: int = Field(default=50, ge=10, le=100)


class ForecastRequest(BaseModel):
    start_date: str
    forecast_days: int = Field(default=60, ge=1, le=365)
    confidence_levels: list[float] = [0.80, 0.90, 0.95]
    model_config: Optional[ForecastConfig] = None


class PredictionPoint(BaseModel):
    date: str
    predicted: float
    lower_90: Optional[float] = None
    upper_90: Optional[float] = None
    lower_80: Optional[float] = None
    upper_80: Optional[float] = None


class ForecastResponse(BaseModel):
    forecast_id: int
    predictions: list[PredictionPoint]
    model_metrics: Optional[dict] = None
    computation_time: Optional[float] = None


class ForecastHistoryResponse(BaseModel):
    forecasts: list[dict]
    total_count: int
    page_info: Optional[dict] = None


# --- LOS Prediction ---

class LOSPredictRequest(BaseModel):
    patient_id: Optional[str] = None
    age_group: str = Field(..., description="e.g., '0-17', '18-35', '36-50', '51-65', '65+'")
    admission_type: str = Field(..., description="Emergency, Elective, Transfer")
    diagnosis_category: str
    admission_day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    admission_hour: Optional[int] = Field(default=None, ge=0, le=23)
    department: str
    is_holiday_period: bool = False


class LOSPredictResponse(BaseModel):
    predicted_los: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    similar_cases: int


class LOSStatisticsResponse(BaseModel):
    mean_los: float
    median_los: float
    p25: float
    p75: float
    p90: float
    distribution: Optional[dict] = None


# --- Model Training ---

class OccupancyTrainRequest(BaseModel):
    training_start: str
    training_end: str
    validation_split: float = Field(default=0.2, ge=0.05, le=0.5)
    hyperparameters: Optional[dict] = None


class LOSTrainRequest(BaseModel):
    training_start: str
    training_end: str
    validation_split: float = Field(default=0.2, ge=0.05, le=0.5)
    model_type: str = "random_forest"


class TrainResponse(BaseModel):
    model_id: str
    training_history: Optional[dict] = None
    metrics: dict
    model_path: Optional[str] = None


class ModelListItem(BaseModel):
    name: str
    version: str
    created_at: str
    metrics: Optional[dict] = None
    is_active: bool = False


class ModelListResponse(BaseModel):
    models: list[ModelListItem]


class ModelMetricsResponse(BaseModel):
    mae: Optional[float] = None
    mape: Optional[float] = None
    rmse: Optional[float] = None
    coverage_90: Optional[float] = None
    interval_width: Optional[float] = None
    last_updated: Optional[str] = None


# --- Analytics ---

class ForecastAccuracyResponse(BaseModel):
    dates: list[str]
    predicted: list[float]
    actual: list[float]
    errors: list[float]
    metrics: dict


class SeasonalPatternsResponse(BaseModel):
    monthly_averages: list[dict]
    day_of_week_patterns: list[dict]
    holiday_impact: Optional[dict] = None
    yearly_trends: list[dict] = []


class CapacityPlanningResponse(BaseModel):
    recommended_beds: int
    utilization_forecast: float
    staffing_recommendations: Optional[dict] = None
    risk_assessment: Optional[dict] = None


class FeatureImportanceResponse(BaseModel):
    occupancy_features: list[dict] = []
    los_features: list[dict] = []
