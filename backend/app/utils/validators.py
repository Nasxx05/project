"""Data validation utilities for uploaded CSV files."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_ADMISSION_COLS = {"patient_id", "admission_date", "discharge_date", "department"}
OPTIONAL_ADMISSION_COLS = {"age_group", "admission_type", "primary_diagnosis_category"}

REQUIRED_HOLIDAY_COLS = {"date", "holiday_name"}
OPTIONAL_HOLIDAY_COLS = {"is_public_holiday", "is_school_holiday", "region"}

VALID_AGE_GROUPS = {"0-17", "18-35", "36-50", "51-65", "65+"}
VALID_ADMISSION_TYPES = {"Emergency", "Elective", "Transfer"}


def validate_admissions_csv(df: pd.DataFrame) -> dict[str, Any]:
    """Validate an admissions CSV DataFrame.

    Returns dict with keys: is_valid, errors, warnings, summary.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check required columns
    missing = REQUIRED_ADMISSION_COLS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")
        return {"is_valid": False, "errors": errors, "warnings": warnings, "summary": {}}

    # Parse dates
    for col in ("admission_date", "discharge_date"):
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            errors.append(f"Cannot parse '{col}' as datetime. Use ISO format (YYYY-MM-DD).")

    if errors:
        return {"is_valid": False, "errors": errors, "warnings": warnings, "summary": {}}

    # Validate discharge >= admission
    bad_dates = df[df["discharge_date"] < df["admission_date"]]
    if len(bad_dates) > 0:
        errors.append(
            f"{len(bad_dates)} records have discharge_date before admission_date."
        )

    # Check for duplicates
    dup_count = df.duplicated(subset=["patient_id", "admission_date"]).sum()
    if dup_count > 0:
        warnings.append(f"{dup_count} potential duplicate records found.")

    # Validate optional columns if present
    if "age_group" in df.columns:
        invalid_ages = set(df["age_group"].dropna().unique()) - VALID_AGE_GROUPS
        if invalid_ages:
            warnings.append(
                f"Non-standard age_group values: {', '.join(sorted(invalid_ages))}. "
                f"Expected: {', '.join(sorted(VALID_AGE_GROUPS))}"
            )

    if "admission_type" in df.columns:
        invalid_types = set(df["admission_type"].dropna().unique()) - VALID_ADMISSION_TYPES
        if invalid_types:
            warnings.append(
                f"Non-standard admission_type values: {', '.join(sorted(invalid_types))}. "
                f"Expected: {', '.join(sorted(VALID_ADMISSION_TYPES))}"
            )

    # Null checks
    for col in REQUIRED_ADMISSION_COLS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null values.")

    summary = {
        "total_records": len(df),
        "date_range": {
            "earliest_admission": str(df["admission_date"].min()),
            "latest_admission": str(df["admission_date"].max()),
        },
        "departments": df["department"].nunique(),
        "unique_patients": df["patient_id"].nunique(),
    }

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def validate_holidays_csv(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a holidays CSV DataFrame.

    Returns dict with keys: is_valid, errors, warnings, summary.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = REQUIRED_HOLIDAY_COLS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")
        return {"is_valid": False, "errors": errors, "warnings": warnings, "summary": {}}

    # Parse date
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        errors.append("Cannot parse 'date' as datetime. Use ISO format (YYYY-MM-DD).")

    if errors:
        return {"is_valid": False, "errors": errors, "warnings": warnings, "summary": {}}

    # Check boolean fields
    for col in ("is_public_holiday", "is_school_holiday"):
        if col in df.columns:
            try:
                df[col] = df[col].astype(bool)
            except Exception:
                warnings.append(f"Column '{col}' contains non-boolean values.")

    # Duplicate dates
    dup_count = df["date"].duplicated().sum()
    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate dates found.")

    summary = {
        "total_records": len(df),
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
        },
    }

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }
