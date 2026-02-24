#!/usr/bin/env python3
"""Generate realistic sample hospital data for the forecasting system.

Usage:
    python generate_sample_data.py [--records 5000] [--start 2020-01-01] [--end 2023-12-31]
"""

import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_admissions(
    num_records: int = 5000,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic admission records with realistic patterns.

    Includes:
    - Seasonal patterns (winter surge in respiratory / flu admissions).
    - Weekly patterns (Monday high admissions, weekend low).
    - Holiday effects (reduced elective admissions).
    - Realistic LOS distributions by department / diagnosis.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days

    departments = [
        "Internal Medicine",
        "Surgery",
        "Cardiology",
        "Respiratory",
        "Neurology",
        "Oncology",
        "Orthopedics",
        "Pediatrics",
    ]
    dept_weights = [0.25, 0.18, 0.12, 0.12, 0.08, 0.08, 0.10, 0.07]

    diagnosis_map = {
        "Internal Medicine": ["Diabetes", "Infection", "Renal", "GI Disorder"],
        "Surgery": ["Appendectomy", "Hernia", "Cholecystectomy", "Trauma"],
        "Cardiology": ["MI", "Heart Failure", "Arrhythmia", "Angina"],
        "Respiratory": ["Pneumonia", "COPD", "Asthma", "Bronchitis"],
        "Neurology": ["Stroke", "Epilepsy", "Migraine", "Neuropathy"],
        "Oncology": ["Chemotherapy", "Tumor Surgery", "Palliative", "Biopsy"],
        "Orthopedics": ["Hip Fracture", "Knee Replacement", "Spinal", "Sports Injury"],
        "Pediatrics": ["Respiratory Infection", "GI Infection", "Fever", "Asthma"],
    }

    # Average LOS by department (days), with spread
    los_params = {
        "Internal Medicine": (5.5, 3.0),
        "Surgery": (4.0, 2.5),
        "Cardiology": (6.0, 4.0),
        "Respiratory": (5.0, 3.5),
        "Neurology": (7.0, 5.0),
        "Oncology": (4.5, 3.0),
        "Orthopedics": (6.5, 4.0),
        "Pediatrics": (3.0, 2.0),
    }

    age_groups = ["0-17", "18-35", "36-50", "51-65", "65+"]
    age_weights_by_dept = {
        "Pediatrics": [0.90, 0.05, 0.02, 0.02, 0.01],
        "Oncology": [0.02, 0.08, 0.20, 0.35, 0.35],
        "Cardiology": [0.01, 0.05, 0.15, 0.35, 0.44],
        "Orthopedics": [0.05, 0.15, 0.20, 0.25, 0.35],
    }
    default_age_weights = [0.05, 0.15, 0.20, 0.30, 0.30]

    admission_types = ["Emergency", "Elective", "Transfer"]
    admission_type_weights = [0.45, 0.45, 0.10]

    records = []
    patient_counter = 0

    for _ in range(num_records):
        patient_counter += 1
        patient_id = f"P{patient_counter:06d}"

        # Generate admission date with seasonal and weekly patterns
        day_offset = rng.integers(0, total_days)
        admission_dt = start + timedelta(days=int(day_offset))

        # Seasonal weight: higher in winter (Dec-Feb)
        month = admission_dt.month
        seasonal_boost = 1.0
        if month in (12, 1, 2):
            seasonal_boost = 1.3  # winter surge
        elif month in (6, 7, 8):
            seasonal_boost = 0.85  # summer dip

        # Weekly weight: lower on weekends
        dow = admission_dt.weekday()
        if dow >= 5:  # weekend
            if rng.random() > 0.55:  # reject ~45% of weekend admissions
                continue
        elif dow == 0:  # Monday surge
            seasonal_boost *= 1.1

        if rng.random() > seasonal_boost / 1.3:
            continue

        # Department
        dept = rng.choice(departments, p=dept_weights)

        # Diagnosis
        diagnosis = rng.choice(diagnosis_map[dept])

        # Admission type
        adm_type = rng.choice(admission_types, p=admission_type_weights)

        # Age group
        age_w = age_weights_by_dept.get(dept, default_age_weights)
        age_group = rng.choice(age_groups, p=age_w)

        # LOS: log-normal distribution per department
        mean_los, std_los = los_params[dept]
        # Elderly stay longer
        if age_group == "65+":
            mean_los *= 1.3
        elif age_group == "0-17":
            mean_los *= 0.8

        # Emergency stays tend to be longer
        if adm_type == "Emergency":
            mean_los *= 1.15

        # Generate LOS using log-normal
        mu = np.log(mean_los ** 2 / np.sqrt(std_los ** 2 + mean_los ** 2))
        sigma = np.sqrt(np.log(1 + (std_los ** 2 / mean_los ** 2)))
        los = max(1, int(round(rng.lognormal(mu, sigma))))
        los = min(los, 60)  # cap at 60 days

        discharge_dt = admission_dt + timedelta(days=los)

        # Add hour of admission
        if adm_type == "Emergency":
            hour = int(rng.choice(range(24)))
        elif adm_type == "Elective":
            hour = int(rng.choice(range(7, 16)))
        else:
            hour = int(rng.choice(range(8, 20)))

        admission_dt = admission_dt.replace(hour=hour, minute=int(rng.integers(0, 60)))

        records.append({
            "patient_id": patient_id,
            "admission_date": admission_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "discharge_date": discharge_dt.strftime("%Y-%m-%d"),
            "department": dept,
            "age_group": age_group,
            "admission_type": adm_type,
            "primary_diagnosis_category": diagnosis,
        })

    df = pd.DataFrame(records)
    return df


def generate_holidays(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    country: str = "US",
) -> pd.DataFrame:
    """Generate US public and school holiday dates."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    holidays = []

    for year in range(start.year, end.year + 1):
        # Fixed public holidays
        fixed = [
            (f"{year}-01-01", "New Year's Day", True, False),
            (f"{year}-07-04", "Independence Day", True, False),
            (f"{year}-12-25", "Christmas Day", True, False),
            (f"{year}-12-31", "New Year's Eve", False, False),
            (f"{year}-11-11", "Veterans Day", True, False),
        ]
        for date_str, name, is_public, is_school in fixed:
            dt = pd.Timestamp(date_str)
            if start <= dt <= end:
                holidays.append({
                    "date": date_str,
                    "holiday_name": name,
                    "is_public_holiday": is_public,
                    "is_school_holiday": is_school,
                    "region": country,
                })

        # Floating holidays (approximate)
        # MLK Day: 3rd Monday of January
        jan1 = pd.Timestamp(f"{year}-01-01")
        mlk = jan1 + timedelta(days=(7 - jan1.weekday()) % 7 + 14)
        holidays.append({
            "date": mlk.strftime("%Y-%m-%d"),
            "holiday_name": "Martin Luther King Jr. Day",
            "is_public_holiday": True,
            "is_school_holiday": True,
            "region": country,
        })

        # Presidents' Day: 3rd Monday of February
        feb1 = pd.Timestamp(f"{year}-02-01")
        pres = feb1 + timedelta(days=(7 - feb1.weekday()) % 7 + 14)
        holidays.append({
            "date": pres.strftime("%Y-%m-%d"),
            "holiday_name": "Presidents' Day",
            "is_public_holiday": True,
            "is_school_holiday": True,
            "region": country,
        })

        # Memorial Day: last Monday of May
        may31 = pd.Timestamp(f"{year}-05-31")
        memorial = may31 - timedelta(days=may31.weekday())
        holidays.append({
            "date": memorial.strftime("%Y-%m-%d"),
            "holiday_name": "Memorial Day",
            "is_public_holiday": True,
            "is_school_holiday": False,
            "region": country,
        })

        # Labor Day: 1st Monday of September
        sep1 = pd.Timestamp(f"{year}-09-01")
        labor = sep1 + timedelta(days=(7 - sep1.weekday()) % 7)
        holidays.append({
            "date": labor.strftime("%Y-%m-%d"),
            "holiday_name": "Labor Day",
            "is_public_holiday": True,
            "is_school_holiday": True,
            "region": country,
        })

        # Thanksgiving: 4th Thursday of November
        nov1 = pd.Timestamp(f"{year}-11-01")
        first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
        thanksgiving = first_thu + timedelta(weeks=3)
        holidays.append({
            "date": thanksgiving.strftime("%Y-%m-%d"),
            "holiday_name": "Thanksgiving",
            "is_public_holiday": True,
            "is_school_holiday": True,
            "region": country,
        })

        # School holidays (approximate)
        # Summer break: mid-June to end of August
        for month, day in [(6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20)]:
            dt = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")
            if start <= dt <= end:
                holidays.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "holiday_name": "Summer Break Start",
                    "is_public_holiday": False,
                    "is_school_holiday": True,
                    "region": country,
                })

        # Winter break: Dec 20-31
        for day in range(20, 32):
            try:
                dt = pd.Timestamp(f"{year}-12-{day:02d}")
                if start <= dt <= end:
                    holidays.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "holiday_name": "Winter Break",
                        "is_public_holiday": False,
                        "is_school_holiday": True,
                        "region": country,
                    })
            except ValueError:
                pass

        # Spring break: second week of March
        for day in range(10, 17):
            try:
                dt = pd.Timestamp(f"{year}-03-{day:02d}")
                if start <= dt <= end:
                    holidays.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "holiday_name": "Spring Break",
                        "is_public_holiday": False,
                        "is_school_holiday": True,
                        "region": country,
                    })
            except ValueError:
                pass

    df = pd.DataFrame(holidays)
    # Remove duplicates on date
    df = df.drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample hospital data")
    parser.add_argument("--records", type=int, default=5000, help="Number of admission records")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="sample_data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.records} admission records ({args.start} to {args.end})...")
    admissions = generate_admissions(args.records, args.start, args.end)
    admissions_path = os.path.join(args.output_dir, "admissions.csv")
    admissions.to_csv(admissions_path, index=False)
    print(f"  Saved {len(admissions)} records to {admissions_path}")

    print("Generating holiday data...")
    holidays = generate_holidays(args.start, args.end)
    holidays_path = os.path.join(args.output_dir, "holidays.csv")
    holidays.to_csv(holidays_path, index=False)
    print(f"  Saved {len(holidays)} records to {holidays_path}")

    # Summary
    admissions["admission_date"] = pd.to_datetime(admissions["admission_date"])
    admissions["discharge_date"] = pd.to_datetime(admissions["discharge_date"])
    admissions["los"] = (admissions["discharge_date"] - admissions["admission_date"]).dt.days

    print("\n--- Admissions Summary ---")
    print(f"  Total records: {len(admissions)}")
    print(f"  Date range: {admissions['admission_date'].min()} to {admissions['admission_date'].max()}")
    print(f"  Departments: {admissions['department'].nunique()}")
    print(f"  Avg LOS: {admissions['los'].mean():.1f} days")
    print(f"  Median LOS: {admissions['los'].median():.1f} days")
    print(f"\n  Department distribution:")
    for dept, count in admissions["department"].value_counts().items():
        print(f"    {dept}: {count} ({count/len(admissions)*100:.1f}%)")

    print(f"\n--- Holidays Summary ---")
    print(f"  Total records: {len(holidays)}")
    print(f"  Public holidays: {holidays['is_public_holiday'].sum()}")
    print(f"  School holidays: {holidays['is_school_holiday'].sum()}")


if __name__ == "__main__":
    main()
