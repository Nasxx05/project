"""Data processing and metrics calculation utilities."""

import numpy as np
import pandas as pd


def detect_outliers(
    series: pd.Series,
    method: str = "IQR",
    threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers in a numeric series.

    Args:
        series: Numeric series to check.
        method: 'IQR' or 'zscore'.
        threshold: Z-score threshold (only for zscore method).

    Returns:
        Boolean mask where True indicates an outlier.
    """
    if method == "IQR":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (series < lower) | (series > upper)
    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    else:
        raise ValueError(f"Unknown method: {method}. Use 'IQR' or 'zscore'.")


def handle_missing_data(
    df: pd.DataFrame,
    strategy: str = "interpolate",
) -> pd.DataFrame:
    """Handle missing values in a DataFrame.

    Args:
        df: Input DataFrame.
        strategy: One of 'interpolate', 'forward_fill', 'mean', 'median'.

    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()
    if strategy == "interpolate":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    elif strategy == "forward_fill":
        df = df.ffill().bfill()
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return df


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    intervals: dict | None = None,
) -> dict:
    """Calculate regression performance metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        intervals: Optional dict with 'lower' and 'upper' arrays for interval metrics.

    Returns:
        Dict of metric name to value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    # Avoid division by zero in MAPE
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(abs_errors[nonzero_mask] / np.abs(y_true[nonzero_mask])) * 100
    else:
        mape = float("nan")

    result = {
        "mae": float(np.mean(abs_errors)),
        "mape": float(mape),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_error": float(np.max(abs_errors)),
    }

    if intervals is not None:
        lower = np.asarray(intervals["lower"], dtype=float)
        upper = np.asarray(intervals["upper"], dtype=float)
        within = (y_true >= lower) & (y_true <= upper)
        result["coverage"] = float(np.mean(within))
        result["avg_interval_width"] = float(np.mean(upper - lower))

    return result


def validate_prediction_intervals(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float,
) -> dict:
    """Check calibration of prediction intervals.

    Returns dict with actual_coverage, is_well_calibrated, interval_stats.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    within = (y_true >= lower) & (y_true <= upper)
    actual_coverage = float(np.mean(within))
    widths = upper - lower

    # Consider well-calibrated if within ±5% of nominal
    is_calibrated = abs(actual_coverage - nominal_coverage) <= 0.05

    return {
        "actual_coverage": actual_coverage,
        "nominal_coverage": nominal_coverage,
        "is_well_calibrated": is_calibrated,
        "interval_stats": {
            "mean_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
            "min_width": float(np.min(widths)),
            "max_width": float(np.max(widths)),
        },
    }
