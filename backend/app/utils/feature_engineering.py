"""Feature engineering utilities for time series and patient data."""

import numpy as np
import pandas as pd


def create_temporal_features(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Extract temporal features from a date series.

    Returns DataFrame with columns for all temporal indicators.
    """
    dates = pd.to_datetime(dates)
    return pd.DataFrame({
        "day_of_week": dates.dayofweek,
        "day_of_year": dates.dayofyear,
        "month": dates.month,
        "week_of_year": dates.isocalendar().week.astype(int).values
            if hasattr(dates, "isocalendar") else dates.isocalendar().week.values,
        "quarter": dates.quarter,
        "is_weekend": (dates.dayofweek >= 5).astype(int),
        "is_monday": (dates.dayofweek == 0).astype(int),
        "is_friday": (dates.dayofweek == 4).astype(int),
        "is_month_start": dates.is_month_start.astype(int),
        "is_month_end": dates.is_month_end.astype(int),
    })


def create_lag_features(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Create lagged versions of a time series.

    Args:
        series: The input time series.
        lags: List of lag values (e.g. [1, 2, 3, 7]).

    Returns:
        DataFrame with columns like 'lag_1', 'lag_2', etc.
    """
    result = pd.DataFrame(index=series.index)
    for lag in lags:
        result[f"lag_{lag}"] = series.shift(lag)
    return result


def create_rolling_features(
    series: pd.Series, windows: list[int]
) -> pd.DataFrame:
    """Create rolling mean and std features.

    Args:
        series: The input time series.
        windows: List of window sizes (e.g. [7, 30]).

    Returns:
        DataFrame with rolling_mean_7, rolling_std_7, etc.
    """
    result = pd.DataFrame(index=series.index)
    for w in windows:
        result[f"rolling_mean_{w}"] = series.rolling(window=w, min_periods=1).mean()
        result[f"rolling_std_{w}"] = series.rolling(window=w, min_periods=1).std().fillna(0)
    return result


def create_holiday_features(
    dates: pd.Series | pd.DatetimeIndex,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create holiday-related features.

    Args:
        dates: Series of dates to create features for.
        holidays_df: DataFrame with columns: date, is_public_holiday, is_school_holiday.

    Returns:
        DataFrame with holiday features.
    """
    dates = pd.to_datetime(dates)
    result = pd.DataFrame(index=range(len(dates)))

    if holidays_df is None or holidays_df.empty:
        result["is_public_holiday"] = 0
        result["is_school_holiday"] = 0
        result["days_since_holiday"] = 30
        result["days_until_holiday"] = 30
        return result

    holidays_df = holidays_df.copy()
    holidays_df["date"] = pd.to_datetime(holidays_df["date"]).dt.normalize()

    public_holidays = set(
        holidays_df.loc[
            holidays_df.get("is_public_holiday", pd.Series(dtype=bool)).fillna(False).astype(bool),
            "date",
        ]
    )
    school_holidays = set(
        holidays_df.loc[
            holidays_df.get("is_school_holiday", pd.Series(dtype=bool)).fillna(False).astype(bool),
            "date",
        ]
    )
    all_holiday_dates = sorted(holidays_df["date"].unique())

    is_public = []
    is_school = []
    days_since = []
    days_until = []

    for d in dates:
        d_norm = pd.Timestamp(d).normalize()
        is_public.append(int(d_norm in public_holidays))
        is_school.append(int(d_norm in school_holidays))

        # Days since last holiday
        past = [h for h in all_holiday_dates if h <= d_norm]
        days_since.append((d_norm - past[-1]).days if past else 30)

        # Days until next holiday
        future = [h for h in all_holiday_dates if h >= d_norm]
        days_until.append((future[0] - d_norm).days if future else 30)

    result["is_public_holiday"] = is_public
    result["is_school_holiday"] = is_school
    result["days_since_holiday"] = days_since
    result["days_until_holiday"] = days_until

    return result


def create_season_feature(months: pd.Series) -> pd.Series:
    """Map month numbers to season labels."""
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    return months.map(season_map)
