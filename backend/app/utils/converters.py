"""Data conversion utilities."""

import pandas as pd
import numpy as np


def admissions_to_occupancy(
    admissions_df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Convert admission records into daily occupancy counts.

    For each date in the range, counts patients where:
        admission_date <= current_date AND discharge_date >= current_date

    Returns DataFrame with columns: [date, occupancy_count].
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start, end=end, freq="D")

    admissions_df = admissions_df.copy()
    admissions_df["admission_date"] = pd.to_datetime(admissions_df["admission_date"])
    admissions_df["discharge_date"] = pd.to_datetime(admissions_df["discharge_date"])

    # Vectorized approach: for each date, count overlapping stays
    counts = []
    for current_date in date_range:
        count = (
            (admissions_df["admission_date"] <= current_date)
            & (admissions_df["discharge_date"] >= current_date)
        ).sum()
        counts.append(count)

    return pd.DataFrame({"date": date_range, "occupancy_count": counts})
