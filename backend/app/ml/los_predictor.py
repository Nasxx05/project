"""Length of Stay (LOS) Predictor using ensemble tree models."""

import logging
import os
from datetime import datetime
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class LOSPredictor:
    """Predict hospital length of stay with prediction intervals."""

    CATEGORICAL_COLS = [
        "age_group",
        "admission_type",
        "department",
        "primary_diagnosis_category",
    ]
    NUMERIC_COLS = [
        "admission_day_of_week",
        "admission_hour",
        "is_weekend_admission",
        "is_night_admission",
        "admission_month",
        "admission_quarter",
        "is_holiday_period",
        "department_avg_los",
        "diagnosis_avg_los",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_type: str = "random_forest"
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Category mappings learned from training data
        self._category_values: dict[str, list[str]] = {}

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(
        self,
        admissions_df: pd.DataFrame,
        holidays_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Create feature matrix from admission records.

        Args:
            admissions_df: DataFrame with admission records (must have discharge_date for training).
            holidays_df: Optional holiday DataFrame.

        Returns:
            Feature DataFrame ready for model input.
        """
        df = admissions_df.copy()
        df["admission_date"] = pd.to_datetime(df["admission_date"])

        # Temporal features from admission date
        df["admission_day_of_week"] = df["admission_date"].dt.dayofweek
        df["admission_hour"] = df["admission_date"].dt.hour
        df["is_weekend_admission"] = (df["admission_date"].dt.dayofweek >= 5).astype(int)
        df["is_night_admission"] = (
            (df["admission_date"].dt.hour >= 18) | (df["admission_date"].dt.hour < 6)
        ).astype(int)
        df["admission_month"] = df["admission_date"].dt.month
        df["admission_quarter"] = df["admission_date"].dt.quarter

        # Holiday period: within 3 days of a holiday
        if holidays_df is not None and not holidays_df.empty:
            hol_dates = set(pd.to_datetime(holidays_df["date"]).dt.normalize())
            df["is_holiday_period"] = df["admission_date"].apply(
                lambda d: int(
                    any(
                        abs((d.normalize() - h).days) <= 3
                        for h in hol_dates
                    )
                )
            )
        else:
            if "is_holiday_period" not in df.columns:
                df["is_holiday_period"] = 0

        # Historical context: rolling averages by department and diagnosis
        if "actual_los" in df.columns:
            dept_avg = df.groupby("department")["actual_los"].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            df["department_avg_los"] = dept_avg.fillna(df["actual_los"].mean())

            if "primary_diagnosis_category" in df.columns:
                diag_avg = df.groupby("primary_diagnosis_category")["actual_los"].transform(
                    lambda x: x.expanding().mean().shift(1)
                )
                df["diagnosis_avg_los"] = diag_avg.fillna(df["actual_los"].mean())
            else:
                df["diagnosis_avg_los"] = df["actual_los"].mean()
        else:
            df["department_avg_los"] = 5.0  # reasonable default
            df["diagnosis_avg_los"] = 5.0

        # Fill missing categoricals
        for col in self.CATEGORICAL_COLS:
            if col not in df.columns:
                df[col] = "Unknown"
            df[col] = df[col].fillna("Unknown").astype(str)

        # Fill missing numerics
        for col in self.NUMERIC_COLS:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def _encode_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Encode categorical and numeric features into a numeric matrix.

        Args:
            df: Feature-engineered DataFrame.
            fit: If True, fit label encoders (training mode).

        Returns:
            Numpy array of encoded features.
        """
        encoded_parts: list[np.ndarray] = []

        for col in self.CATEGORICAL_COLS:
            if col not in df.columns:
                encoded_parts.append(np.zeros((len(df), 1)))
                continue
            if fit:
                le = LabelEncoder()
                le.fit(df[col])
                self.label_encoders[col] = le
                self._category_values[col] = list(le.classes_)
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    encoded_parts.append(np.zeros((len(df), 1)))
                    continue
                # Handle unseen categories
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            encoded_parts.append(le.transform(df[col]).reshape(-1, 1))

        numeric = df[self.NUMERIC_COLS].values
        encoded_parts.append(numeric)

        X = np.hstack(encoded_parts)

        if fit:
            X = self.scaler.fit_transform(X)
            self.feature_names = self.CATEGORICAL_COLS + self.NUMERIC_COLS
        else:
            X = self.scaler.transform(X)

        return X

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self, model_type: str = "random_forest"):
        """Build the LOS prediction model.

        Args:
            model_type: 'random_forest' or 'gradient_boosting'.
        """
        self.model_type = model_type
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        admissions_df: pd.DataFrame,
        holidays_df: Optional[pd.DataFrame] = None,
        validation_split: float = 0.2,
        model_type: str = "random_forest",
    ) -> dict[str, Any]:
        """Train the LOS predictor.

        Args:
            admissions_df: Admission data with actual_los column.
            holidays_df: Holiday data.
            validation_split: Fraction of data for validation.
            model_type: 'random_forest' or 'gradient_boosting'.

        Returns:
            Dict with metrics, feature_importance, training_info.
        """
        # Filter to discharged patients with known LOS
        df = admissions_df.copy()
        df["admission_date"] = pd.to_datetime(df["admission_date"])
        df["discharge_date"] = pd.to_datetime(df["discharge_date"])
        df["actual_los"] = (df["discharge_date"] - df["admission_date"]).dt.days
        df = df[df["actual_los"] >= 0].copy()

        logger.info("Training LOS model on %d records", len(df))

        feature_df = self.engineer_features(df, holidays_df)
        X = self._encode_features(feature_df, fit=True)
        y = df["actual_los"].values.astype(float)

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Build and train
        self.build_model(model_type)
        self.model.fit(X_train, y_train)

        # Evaluate
        val_pred = self.model.predict(X_val)
        from ..utils.data_processing import calculate_metrics

        metrics = calculate_metrics(y_val, val_pred)

        # Feature importance
        importances = self.model.feature_importances_
        feature_importance = [
            {"feature_name": name, "importance_score": float(imp), "rank": rank + 1}
            for rank, (name, imp) in enumerate(
                sorted(zip(self.feature_names, importances), key=lambda x: -x[1])
            )
        ]

        self.model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        return {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "training_info": {
                "model_type": model_type,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "n_features": X_train.shape[1],
            },
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_with_intervals(
        self,
        patient_features: dict | pd.DataFrame,
        confidence_level: float = 0.90,
    ) -> dict[str, Any]:
        """Predict LOS with prediction intervals.

        For tree-based models, uses individual tree predictions to form intervals.
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")

        # Prepare input
        if isinstance(patient_features, dict):
            df = pd.DataFrame([patient_features])
        else:
            df = patient_features.copy()

        feature_df = self.engineer_features(df)
        X = self._encode_features(feature_df, fit=False)

        # Get predictions from individual estimators
        if hasattr(self.model, "estimators_"):
            # RandomForest or GradientBoosting ensemble
            if self.model_type == "random_forest":
                tree_preds = np.array([
                    tree.predict(X).ravel() for tree in self.model.estimators_
                ])
            else:
                # GradientBoosting: use staged_predict
                tree_preds = np.array([
                    pred.ravel() for pred in self.model.staged_predict(X)
                ])

            mean_pred = np.mean(tree_preds, axis=0)
            alpha = 1 - confidence_level
            lower = np.percentile(tree_preds, alpha / 2 * 100, axis=0)
            upper = np.percentile(tree_preds, (1 - alpha / 2) * 100, axis=0)
        else:
            mean_pred = self.model.predict(X)
            # Fallback: use ±20% as rough interval
            lower = mean_pred * 0.8
            upper = mean_pred * 1.2

        # Ensure non-negative
        mean_pred = np.maximum(mean_pred, 0)
        lower = np.maximum(lower, 0)

        if len(mean_pred) == 1:
            return {
                "predicted_los": float(round(mean_pred[0], 1)),
                "lower_bound": float(round(lower[0], 1)),
                "upper_bound": float(round(upper[0], 1)),
                "confidence_level": confidence_level,
            }

        return {
            "predictions": [
                {
                    "predicted_los": float(round(m, 1)),
                    "lower_bound": float(round(lo, 1)),
                    "upper_bound": float(round(hi, 1)),
                }
                for m, lo, hi in zip(mean_pred, lower, upper)
            ],
            "confidence_level": confidence_level,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        admissions_df: pd.DataFrame,
        holidays_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Evaluate on test data."""
        if self.model is None:
            raise RuntimeError("No model loaded.")

        df = admissions_df.copy()
        df["admission_date"] = pd.to_datetime(df["admission_date"])
        df["discharge_date"] = pd.to_datetime(df["discharge_date"])
        df["actual_los"] = (df["discharge_date"] - df["admission_date"]).dt.days
        df = df[df["actual_los"] >= 0].copy()

        feature_df = self.engineer_features(df, holidays_df)
        X = self._encode_features(feature_df, fit=False)
        y = df["actual_los"].values.astype(float)

        pred = self.model.predict(X)
        from ..utils.data_processing import calculate_metrics

        return calculate_metrics(y, pred)

    def get_feature_importance(self) -> list[dict]:
        """Return ranked feature importances."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return []

        importances = self.model.feature_importances_
        return [
            {"feature_name": name, "importance_score": float(imp), "rank": rank + 1}
            for rank, (name, imp) in enumerate(
                sorted(zip(self.feature_names, importances), key=lambda x: -x[1])
            )
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save model and preprocessing objects."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, "los_model.joblib"))
        joblib.dump(self.scaler, os.path.join(path, "los_scaler.joblib"))
        joblib.dump(self.label_encoders, os.path.join(path, "los_encoders.joblib"))
        joblib.dump(
            {
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "model_version": self.model_version,
                "category_values": self._category_values,
            },
            os.path.join(path, "los_metadata.joblib"),
        )
        logger.info("LOS model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load model and preprocessing objects."""
        self.model = joblib.load(os.path.join(path, "los_model.joblib"))
        self.scaler = joblib.load(os.path.join(path, "los_scaler.joblib"))
        self.label_encoders = joblib.load(os.path.join(path, "los_encoders.joblib"))
        meta = joblib.load(os.path.join(path, "los_metadata.joblib"))
        self.feature_names = meta["feature_names"]
        self.model_type = meta["model_type"]
        self.model_version = meta["model_version"]
        self._category_values = meta.get("category_values", {})
        logger.info("LOS model loaded from %s (version %s)", path, self.model_version)
