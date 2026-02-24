"""NARX (Nonlinear AutoRegressive with eXogenous inputs) Bed Occupancy Forecaster.

Implements bootstrap ensemble prediction intervals following Kutafina et al. (2019).
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# TensorFlow import with graceful fallback
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available; NARX model will not function.")


class NARXOccupancyForecaster:
    """NARX neural network for bed occupancy forecasting with prediction intervals."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.config = config or {}
        self.delay = self.config.get("delay", 2)
        self.hidden_layers = self.config.get("hidden_layers", 2)
        self.nodes_per_layer = self.config.get("nodes_per_layer", 10)
        self.epochs = self.config.get("epochs", 1000)
        self.learning_rate = self.config.get("learning_rate", 0.001)

        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self.model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(
        self,
        occupancy_df: pd.DataFrame,
        holidays_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build full feature matrix from occupancy and holiday data.

        Args:
            occupancy_df: DataFrame with columns [date, occupancy_count].
            holidays_df: Optional DataFrame with holiday information.

        Returns:
            DataFrame with all engineered features aligned to occupancy dates.
        """
        df = occupancy_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # --- Historical occupancy lags ---
        for lag in range(1, 8):
            df[f"occ_lag_{lag}"] = df["occupancy_count"].shift(lag)

        # --- Temporal features ---
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_year"] = df["date"].dt.dayofyear
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int).values
        df["quarter"] = df["date"].dt.quarter
        df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
        df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
        df["is_friday"] = (df["date"].dt.dayofweek == 4).astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

        # --- Rolling features ---
        df["rolling_mean_7"] = (
            df["occupancy_count"].rolling(window=7, min_periods=1).mean()
        )
        df["rolling_mean_30"] = (
            df["occupancy_count"].rolling(window=30, min_periods=1).mean()
        )
        df["rolling_std_7"] = (
            df["occupancy_count"].rolling(window=7, min_periods=1).std().fillna(0)
        )
        df["occupancy_trend"] = df["occupancy_count"] - df["rolling_mean_7"]

        # --- Holiday features ---
        if holidays_df is not None and not holidays_df.empty:
            hdf = holidays_df.copy()
            hdf["date"] = pd.to_datetime(hdf["date"]).dt.normalize()
            date_set = set(df["date"].dt.normalize())

            public_set = set(
                hdf.loc[hdf.get("is_public_holiday", pd.Series(dtype=bool)).fillna(False).astype(bool), "date"]
            )
            school_set = set(
                hdf.loc[hdf.get("is_school_holiday", pd.Series(dtype=bool)).fillna(False).astype(bool), "date"]
            )
            all_hol = sorted(hdf["date"].unique())

            df["is_public_holiday"] = df["date"].dt.normalize().isin(public_set).astype(int)
            df["is_school_holiday"] = df["date"].dt.normalize().isin(school_set).astype(int)

            days_since = []
            days_until = []
            for d in df["date"].dt.normalize():
                past = [h for h in all_hol if h <= d]
                days_since.append((d - past[-1]).days if past else 30)
                future = [h for h in all_hol if h >= d]
                days_until.append((future[0] - d).days if future else 30)
            df["days_since_holiday"] = days_since
            df["days_until_holiday"] = days_until
        else:
            df["is_public_holiday"] = 0
            df["is_school_holiday"] = 0
            df["days_since_holiday"] = 30
            df["days_until_holiday"] = 30

        # --- Seasonal encoding (sin/cos for cyclical features) ---
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Drop rows with NaN from lagging
        df = df.dropna().reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Sequence creation for NARX
    # ------------------------------------------------------------------

    def create_sequences(
        self, feature_df: pd.DataFrame, target_col: str = "occupancy_count"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create NARX input sequences using the configured delay.

        For delay=2, each sample contains features from t-2, t-1, t.
        """
        exclude_cols = {"date", target_col}
        feature_cols = [c for c in feature_df.columns if c not in exclude_cols]
        self.feature_columns = feature_cols

        features = feature_df[feature_cols].values
        targets = feature_df[target_col].values

        X, y = [], []
        for i in range(self.delay, len(features)):
            # Flatten delay window of features into a single vector
            seq = features[i - self.delay: i + 1].flatten()
            X.append(seq)
            y.append(targets[i])

        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------

    def build_model(self, input_dim: int) -> "keras.Model":
        """Build the NARX neural network."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required to build the NARX model.")

        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        for i in range(self.hidden_layers):
            model.add(layers.Dense(self.nodes_per_layer, activation="relu"))
            model.add(layers.Dropout(0.2))

        # Optional third hidden layer
        if self.config.get("extra_layer", False):
            model.add(layers.Dense(self.nodes_per_layer, activation="relu"))

        model.add(layers.Dense(1))

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        occupancy_df: pd.DataFrame,
        holidays_df: Optional[pd.DataFrame] = None,
        validation_split: float = 0.2,
        early_stopping_patience: int = 50,
    ) -> dict[str, Any]:
        """Train the NARX model on occupancy data.

        Uses temporal split (not random) for validation.
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for training.")

        logger.info("Engineering features...")
        feature_df = self.engineer_features(occupancy_df, holidays_df)

        logger.info("Creating sequences (delay=%d)...", self.delay)
        X, y = self.create_sequences(feature_df)

        # Temporal train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).ravel()

        logger.info(
            "Training NARX model: %d train, %d val samples, input_dim=%d",
            len(X_train), len(X_val), X_train.shape[1],
        )

        self.model = self.build_model(X_train.shape[1])

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
        ]

        history = self.model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=self.epochs,
            batch_size=32,
            callbacks=cb,
            verbose=0,
        )

        # Evaluate on validation set
        val_pred_scaled = self.model.predict(X_val, verbose=0).ravel()
        val_pred = self.target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()

        from ..utils.data_processing import calculate_metrics
        metrics = calculate_metrics(y_val, val_pred)

        self.model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        return {
            "training_history": {
                "loss": [float(v) for v in history.history["loss"]],
                "val_loss": [float(v) for v in history.history["val_loss"]],
                "mae": [float(v) for v in history.history["mae"]],
                "val_mae": [float(v) for v in history.history["val_mae"]],
            },
            "metrics": metrics,
            "epochs_trained": len(history.history["loss"]),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
        }

    # ------------------------------------------------------------------
    # Prediction with bootstrap intervals
    # ------------------------------------------------------------------

    def predict_with_intervals(
        self,
        occupancy_df: pd.DataFrame,
        holidays_df: Optional[pd.DataFrame] = None,
        forecast_days: int = 60,
        confidence_levels: list[float] | None = None,
        n_bootstrap: int = 50,
    ) -> dict[str, Any]:
        """Generate forecasts with prediction intervals using bootstrap ensemble.

        Steps:
        1. Engineer features from historical data.
        2. For each bootstrap iteration, add noise to residuals and re-predict
           iteratively for each forecast day.
        3. Aggregate bootstrap predictions into mean + percentile intervals.
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load_model() first.")

        if confidence_levels is None:
            confidence_levels = [0.80, 0.90, 0.95]

        start_time = time.time()
        feature_df = self.engineer_features(occupancy_df, holidays_df)

        X_all, y_all = self.create_sequences(feature_df)
        X_scaled = self.scaler.transform(X_all)

        # Calculate residuals on training data for noise estimation
        pred_scaled = self.model.predict(X_scaled, verbose=0).ravel()
        pred_all = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        residuals = y_all - pred_all
        residual_std = float(np.std(residuals))

        # Get last date in the data
        last_date = pd.to_datetime(feature_df["date"].iloc[-1])
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

        # Bootstrap ensemble predictions
        all_predictions = np.zeros((n_bootstrap, forecast_days))

        # Base prediction: iterative multi-step forecast
        # We need to roll forward, using predicted values as input for the next step
        n_features = len(self.feature_columns)

        for b in range(n_bootstrap):
            # Start from the end of the historical feature matrix
            # Add bootstrap noise to simulate uncertainty
            recent_features = feature_df.copy()
            recent_occupancy = occupancy_df.copy()

            for d in range(forecast_days):
                target_date = forecast_dates[d]

                # Append the predicted occupancy to the rolling history
                if d > 0:
                    new_row = pd.DataFrame({
                        "date": [target_date - timedelta(days=1)],
                        "occupancy_count": [all_predictions[b, d - 1]],
                    })
                    recent_occupancy = pd.concat(
                        [recent_occupancy, new_row], ignore_index=True
                    )

                # Re-engineer features with updated history
                updated_features = self.engineer_features(recent_occupancy, holidays_df)
                if len(updated_features) < self.delay + 1:
                    # Not enough history, use last known value
                    all_predictions[b, d] = pred_all[-1] + np.random.normal(0, residual_std)
                    continue

                X_last, _ = self.create_sequences(updated_features)
                if len(X_last) == 0:
                    all_predictions[b, d] = pred_all[-1] + np.random.normal(0, residual_std)
                    continue

                X_last_scaled = self.scaler.transform(X_last[-1:])
                pred_scaled_val = self.model.predict(X_last_scaled, verbose=0).ravel()[0]
                pred_val = self.target_scaler.inverse_transform(
                    np.array([[pred_scaled_val]])
                ).ravel()[0]

                # Add bootstrap noise
                noise = np.random.normal(0, residual_std)
                all_predictions[b, d] = max(0, pred_val + noise)

        # Aggregate predictions
        mean_predictions = np.mean(all_predictions, axis=0)
        intervals: dict[str, dict[str, list[float]]] = {}

        for cl in confidence_levels:
            alpha = 1 - cl
            lower_pct = alpha / 2 * 100
            upper_pct = (1 - alpha / 2) * 100
            key = f"{int(cl * 100)}"
            intervals[key] = {
                "lower": np.percentile(all_predictions, lower_pct, axis=0).tolist(),
                "upper": np.percentile(all_predictions, upper_pct, axis=0).tolist(),
            }

        computation_time = time.time() - start_time

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "predictions": mean_predictions.tolist(),
            "intervals": intervals,
            "computation_time": computation_time,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, occupancy_df: pd.DataFrame, holidays_df: Optional[pd.DataFrame] = None
    ) -> dict[str, float]:
        """Evaluate model on test data."""
        if self.model is None:
            raise RuntimeError("No model loaded.")

        feature_df = self.engineer_features(occupancy_df, holidays_df)
        X, y = self.create_sequences(feature_df)
        X_scaled = self.scaler.transform(X)

        pred_scaled = self.model.predict(X_scaled, verbose=0).ravel()
        pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

        from ..utils.data_processing import calculate_metrics
        return calculate_metrics(y, pred)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save model weights, scalers, and config."""
        os.makedirs(path, exist_ok=True)
        if self.model is not None:
            self.model.save(os.path.join(path, "narx_model.keras"))
        joblib.dump(self.scaler, os.path.join(path, "feature_scaler.joblib"))
        joblib.dump(self.target_scaler, os.path.join(path, "target_scaler.joblib"))
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "config": self.config,
                "delay": self.delay,
                "model_version": self.model_version,
            },
            os.path.join(path, "metadata.joblib"),
        )
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load model weights, scalers, and config."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required to load the NARX model.")

        model_file = os.path.join(path, "narx_model.keras")
        if os.path.exists(model_file):
            self.model = keras.models.load_model(model_file)
        self.scaler = joblib.load(os.path.join(path, "feature_scaler.joblib"))
        self.target_scaler = joblib.load(os.path.join(path, "target_scaler.joblib"))
        meta = joblib.load(os.path.join(path, "metadata.joblib"))
        self.feature_columns = meta["feature_columns"]
        self.config = meta["config"]
        self.delay = meta["delay"]
        self.model_version = meta["model_version"]
        logger.info("Model loaded from %s (version %s)", path, self.model_version)
