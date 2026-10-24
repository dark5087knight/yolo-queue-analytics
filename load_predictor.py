"""
LightGBM Queue Load & Wait Time Prediction Engine.

Uses historical customer traffic logs extracted by the YOLOv8 + DeepSORT queue monitoring
system (Customer ID, Waiting Time, Hour, Day, Zone, Entry Time) to train LightGBM gradient
boosting models that predict:
  1. Expected Customer Load (traffic volume / customer count) for any day & hour.
  2. Expected Wait Time (load duration in seconds) for any day, hour, and queue load.
  3. Peak traffic hours, operational status classification, and 24-hour load forecasting.
"""

import os
import sys
import json
import math
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_TO_INT = {name: i for i, name in enumerate(DAY_NAMES)}
INT_TO_DAY = {i: name for i, name in enumerate(DAY_NAMES)}


def time_of_day_bin(hour: int) -> int:
    """Bins 24 hours into operational periods:
    0: Night/Closed (22:00 - 05:59)
    1: Early Morning (06:00 - 08:59)
    2: Morning Peak (09:00 - 11:59)
    3: Lunch Rush (12:00 - 14:59)
    4: Afternoon (15:00 - 17:59)
    5: Evening Peak (18:00 - 21:59)
    """
    if hour < 6 or hour >= 22:
        return 0
    elif 6 <= hour < 9:
        return 1
    elif 9 <= hour < 12:
        return 2
    elif 12 <= hour < 15:
        return 3
    elif 15 <= hour < 18:
        return 4
    else:
        return 5


def generate_synthetic_historical_data(
    output_path: str,
    days: int = 28,
    start_date: str = "2026-03-23"
) -> pd.DataFrame:
    """
    Generates a realistic 4-week historical customer visit dataset matching the exact
    format produced by the YOLOv8 queue monitoring report window:
    Columns: Customer ID, Waiting Time, Hour, Day, Zone, Entry Time, Exit Time, Status
    """
    np.random.seed(42)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    records = []
    customer_id = 1

    # Base arrival rates (Poisson lambda) per hour across typical business day (8:00 - 20:00)
    hourly_arrival_weights = {
        8: 14, 9: 22, 10: 18, 11: 15,
        12: 25, 13: 28, 14: 16, 15: 12,
        16: 15, 17: 24, 18: 20, 19: 11,
    }

    # Day-of-week multipliers (e.g., Mondays & Fridays busier)
    day_multipliers = {
        "Monday": 1.25,
        "Tuesday": 1.0,
        "Wednesday": 0.95,
        "Thursday": 1.1,
        "Friday": 1.35,
        "Saturday": 0.7,
        "Sunday": 0.5,
    }

    for d in range(days):
        current_day_dt = start_dt + timedelta(days=d)
        day_name = current_day_dt.strftime("%A")
        day_mult = day_multipliers.get(day_name, 1.0)

        for hour, base_lambda in hourly_arrival_weights.items():
            expected_arrivals = max(1, int(round(base_lambda * day_mult + np.random.normal(0, 2))))
            
            for _ in range(expected_arrivals):
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)
                entry_time = current_day_dt.replace(hour=hour, minute=minute, second=second)

                # Base wait time influenced by current hourly queue load:
                # Higher load -> higher wait time with realistic variance
                base_wait = 8.0 + (expected_arrivals * 0.55) + np.random.exponential(scale=3.5)
                # Zone 1 vs Zone 2
                zone = 1 if np.random.rand() > 0.3 else 2
                if zone == 2:
                    base_wait *= 1.15  # Zone 2 slightly longer

                wait_sec = round(max(3.0, float(base_wait)), 1)
                exit_time = entry_time + timedelta(seconds=wait_sec)

                records.append({
                    "Customer ID": customer_id,
                    "Waiting Time": wait_sec,
                    "Hour": hour,
                    "Day": day_name,
                    "Zone": zone,
                    "Entry Time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Exit Time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Status": "Served",
                })
                customer_id += 1

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated realistic dataset with {len(df)} customer logs -> {output_path}")
    return df


class QueueDataProcessor:
    """Preprocesses raw queue CSV files and engineers features for LightGBM."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_hourly: Optional[pd.DataFrame] = None
        self.df_individual: Optional[pd.DataFrame] = None

    def load_and_clean(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Queue data file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Standardize column names
        rename_map = {}
        for col in df.columns:
            clean = col.strip()
            if clean in ["Waiting Time", "Wait (Seconds)", "Wait (s)", "wait_seconds"]:
                rename_map[col] = "waiting_time"
            elif clean in ["Hour", "hour"]:
                rename_map[col] = "hour"
            elif clean in ["Day", "day"]:
                rename_map[col] = "day"
            elif clean in ["Zone", "zone"]:
                rename_map[col] = "zone"
            elif clean in ["Customer ID", "customer_id", "ID", "id"]:
                rename_map[col] = "customer_id"
            elif clean in ["Entry Time", "entry_time"]:
                rename_map[col] = "entry_time"
            elif clean in ["Exit Time", "exit_time"]:
                rename_map[col] = "exit_time"
            elif clean in ["Status", "status"]:
                rename_map[col] = "status"

        df = df.rename(columns=rename_map)

        if "waiting_time" not in df.columns or "hour" not in df.columns:
            raise ValueError(f"CSV must contain waiting time and hour columns. Found: {list(df.columns)}")

        # Clean numerical values
        df["waiting_time"] = pd.to_numeric(df["waiting_time"], errors="coerce")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(12).astype(int)
        df["zone"] = pd.to_numeric(df.get("zone", 1), errors="coerce").fillna(1).astype(int)

        # Handle missing days
        if "day" not in df.columns:
            if "entry_time" in df.columns:
                df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
                df["day"] = df["entry_time"].dt.day_name()
            else:
                df["day"] = "Monday"
        df["day"] = df["day"].astype(str).str.strip().str.capitalize()

        # Drop invalid rows
        df = df.dropna(subset=["waiting_time", "hour"])
        df = df[(df["waiting_time"] > 0) & (df["waiting_time"] < 7200)]  # Filter realistic waits (< 2h)
        df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]

        self.df_raw = df
        return df

    def build_feature_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds two analytical datasets:
        1. Hourly Aggregated Dataset: Models queue load (customer count) per (Day, Hour, Date/Zone).
        2. Individual Customer Dataset: Models wait duration (seconds) with load and temporal features.
        """
        if self.df_raw is None:
            self.load_and_clean()

        df = self.df_raw.copy()

        # Extract date string if entry_time exists
        if "entry_time" in df.columns:
            df["datetime"] = pd.to_datetime(df["entry_time"], errors="coerce")
            df["date"] = df["datetime"].dt.date
            df["minute"] = df["datetime"].dt.minute.fillna(0).astype(int)
        else:
            df["date"] = "2026-04-20"
            df["minute"] = 0

        # 1. Hourly Aggregated Dataset for Queue Load
        group_cols = ["date", "day", "hour", "zone"]
        hourly_agg = df.groupby(group_cols).agg(
            customer_load=("customer_id", "count"),
            avg_wait_time=("waiting_time", "mean"),
            p90_wait_time=("waiting_time", lambda x: np.percentile(x, 90)),
        ).reset_index()

        hourly_features = self._engineer_temporal_features(hourly_agg)
        self.df_hourly = hourly_features

        # 2. Individual Customer Dataset for Wait Duration
        # Merge hourly load into individual visits as a feature
        df_indiv = df.merge(
            hourly_agg[["date", "day", "hour", "zone", "customer_load"]],
            on=["date", "day", "hour", "zone"],
            how="left"
        )
        df_indiv["customer_load"] = df_indiv["customer_load"].fillna(1)
        indiv_features = self._engineer_temporal_features(df_indiv)
        self.df_individual = indiv_features

        return hourly_features, indiv_features

    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies ML Best Practice feature engineering for tabular/temporal data."""
        data = df.copy()

        # Day of week mapping
        data["day_num"] = data["day"].map(DAY_TO_INT).fillna(0).astype(int)
        data["is_weekend"] = data["day_num"].isin([5, 6]).astype(int)

        # Hour features
        data["hour"] = data["hour"].astype(int)
        data["time_bin"] = data["hour"].apply(time_of_day_bin)

        # Cyclical Encodings (smooth 24-hour and 7-day periodicity)
        data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24.0)
        data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24.0)
        data["sin_day"] = np.sin(2 * np.pi * data["day_num"] / 7.0)
        data["cos_day"] = np.cos(2 * np.pi * data["day_num"] / 7.0)

        # Interaction features
        data["peak_rush_interaction"] = (
            ((data["hour"].isin([8, 9, 12, 13, 17, 18])) & (data["is_weekend"] == 0))
        ).astype(int)

        return data


class LightGBMQueueTrainer:
    """
    Trains and evaluates LightGBM regressors for:
    - Queue Load (customer volume / traffic intensity)
    - Queue Wait Time (load duration in seconds)
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

        self.load_model: Optional[lgb.LGBMRegressor] = None
        self.wait_model: Optional[lgb.LGBMRegressor] = None

        self.load_model_path = os.path.join(self.models_dir, "lightgbm_load_model.txt")
        self.wait_model_path = os.path.join(self.models_dir, "lightgbm_wait_model.txt")
        self.meta_path = os.path.join(self.models_dir, "load_model_meta.json")

        self.load_feature_names: List[str] = [
            "day_num", "hour", "zone", "is_weekend", "time_bin",
            "sin_hour", "cos_hour", "sin_day", "cos_day", "peak_rush_interaction"
        ]

        self.wait_feature_names: List[str] = [
            "day_num", "hour", "zone", "customer_load", "is_weekend",
            "time_bin", "sin_hour", "cos_hour", "sin_day", "cos_day", "peak_rush_interaction"
        ]

    def train_models(
        self,
        df_hourly: pd.DataFrame,
        df_individual: pd.DataFrame,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Trains both LightGBM models following ML best practices:
        - Chronological train/validation/test split
        - Early stopping
        - Evaluation on unseen test split (MAE, RMSE, R²)
        """
        results = {}

        # -------------------------------------------------------------
        # 1. Train Queue Load Model (Customer Arrivals / Count per Hour)
        # -------------------------------------------------------------
        print("\n" + "=" * 60)
        print(" [1/2] Training LightGBM Model: Queue Load (Traffic Count)")
        print("=" * 60)

        X_load = df_hourly[self.load_feature_names]
        y_load = df_hourly["customer_load"].values

        n_samples = len(X_load)
        if n_samples < 10:
            raise ValueError(f"Insufficient hourly data points ({n_samples}). Provide at least 15+ samples.")

        # Chronological Split: 70% Train, 15% Validation, 15% Test
        train_idx = int(0.70 * n_samples)
        val_idx = int(0.85 * n_samples)

        X_train_l, y_train_l = X_load.iloc[:train_idx], y_load[:train_idx]
        X_val_l, y_val_l = X_load.iloc[train_idx:val_idx], y_load[train_idx:val_idx]
        X_test_l, y_test_l = X_load.iloc[val_idx:], y_load[val_idx:]

        self.load_model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=20,
            max_depth=5,
            min_child_samples=max(2, int(len(X_train_l) * 0.05)),
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            objective="regression",
            verbose=-1,
        )

        self.load_model.fit(
            X_train_l, y_train_l,
            eval_set=[(X_val_l, y_val_l)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        # Test Evaluation
        y_pred_l = self.load_model.predict(X_test_l)
        y_pred_l = np.clip(y_pred_l, a_min=0, a_max=None)

        mae_load = float(mean_absolute_error(y_test_l, y_pred_l))
        rmse_load = float(math.sqrt(mean_squared_error(y_test_l, y_pred_l)))
        r2_load = float(r2_score(y_test_l, y_pred_l)) if len(y_test_l) > 1 else 1.0

        print(f"Test MAE:   {mae_load:.3f} customers/hour")
        print(f"Test RMSE:  {rmse_load:.3f} customers/hour")
        print(f"Test R²:    {r2_load:.3f}")

        # Feature importances
        load_importances = dict(zip(self.load_feature_names, [float(x) for x in self.load_model.feature_importances_]))
        sorted_imp_l = sorted(load_importances.items(), key=lambda x: x[1], reverse=True)
        print("Top Features (Load):", ", ".join([f"{k} ({v:.1f})" for k, v in sorted_imp_l[:4]]))

        # -------------------------------------------------------------
        # 2. Train Load Time Model (Waiting Duration in Seconds)
        # -------------------------------------------------------------
        print("\n" + "=" * 60)
        print(" [2/2] Training LightGBM Model: Load Time (Wait Duration)")
        print("=" * 60)

        X_wait = df_individual[self.wait_feature_names]
        y_wait = df_individual["waiting_time"].values

        n_wait = len(X_wait)
        train_idx_w = int(0.70 * n_wait)
        val_idx_w = int(0.85 * n_wait)

        X_train_w, y_train_w = X_wait.iloc[:train_idx_w], y_wait[:train_idx_w]
        X_val_w, y_val_w = X_wait.iloc[train_idx_w:val_idx_w], y_wait[train_idx_w:val_idx_w]
        X_test_w, y_test_w = X_wait.iloc[val_idx_w:], y_wait[val_idx_w:]

        self.wait_model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=31,
            max_depth=6,
            min_child_samples=max(3, int(len(X_train_w) * 0.03)),
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            objective="huber",  # Robust against extreme wait outliers
            verbose=-1,
        )

        self.wait_model.fit(
            X_train_w, y_train_w,
            eval_set=[(X_val_w, y_val_w)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        y_pred_w = self.wait_model.predict(X_test_w)
        y_pred_w = np.clip(y_pred_w, a_min=1.0, a_max=None)

        mae_wait = float(mean_absolute_error(y_test_w, y_pred_w))
        rmse_wait = float(math.sqrt(mean_squared_error(y_test_w, y_pred_w)))
        r2_wait = float(r2_score(y_test_w, y_pred_w)) if len(y_test_w) > 1 else 1.0

        print(f"Test MAE:   {mae_wait:.2f} seconds")
        print(f"Test RMSE:  {rmse_wait:.2f} seconds")
        print(f"Test R²:    {r2_wait:.3f}")

        wait_importances = dict(zip(self.wait_feature_names, [float(x) for x in self.wait_model.feature_importances_]))
        sorted_imp_w = sorted(wait_importances.items(), key=lambda x: x[1], reverse=True)
        print("Top Features (Wait):", ", ".join([f"{k} ({v:.1f})" for k, v in sorted_imp_w[:4]]))

        # Calibrated operational stats
        calib_avg_wait = round(float(np.mean(y_wait)), 2)
        peak_hour = int(df_hourly.groupby("hour")["customer_load"].mean().idxmax())

        # Save models to file
        self.load_model.booster_.save_model(self.load_model_path)
        self.wait_model.booster_.save_model(self.wait_model_path)

        meta = {
            "trained_at": datetime.now().isoformat(),
            "records_trained": int(len(df_individual)),
            "hourly_samples": int(len(df_hourly)),
            "load_metrics": {"mae": round(mae_load, 3), "rmse": round(rmse_load, 3), "r2": round(r2_load, 3)},
            "wait_metrics": {"mae": round(mae_wait, 2), "rmse": round(rmse_wait, 2), "r2": round(r2_wait, 3)},
            "top_features_load": dict(sorted_imp_l[:5]),
            "top_features_wait": dict(sorted_imp_w[:5]),
            "calibrated_peak_hour": peak_hour,
            "calibrated_avg_wait_sec": calib_avg_wait,
            "load_model_path": self.load_model_path,
            "wait_model_path": self.wait_model_path,
        }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

        print("\n" + "=" * 60)
        print(f"Models successfully saved to:\n • {self.load_model_path}\n • {self.wait_model_path}")
        print(f"Metadata saved to: {self.meta_path}")
        print("=" * 60)

        results = {
            "meta": meta,
            "calib_avg_wait": calib_avg_wait,
            "peak_hour": peak_hour,
        }
        return results

    def update_config(self, avg_wait: float, peak_hour: int):
        """Updates config.py directly to align GUI and queue manager parameters."""
        config_path = os.path.join(PROJECT_ROOT, "config.py")
        if not os.path.exists(config_path):
            return

        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = re.sub(
            r"AVERAGE_WAIT_TIME_SECONDS\s*=\s*[\d\.]+",
            f"AVERAGE_WAIT_TIME_SECONDS = {avg_wait}",
            content
        )
        content = re.sub(
            r"PEAK_HOUR\s*=\s*\d+",
            f"PEAK_HOUR = {peak_hour}",
            content
        )

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)

        config.AVERAGE_WAIT_TIME_SECONDS = avg_wait
        config.PEAK_HOUR = peak_hour
        print(f"Synchronized config.py: AVERAGE_WAIT_TIME_SECONDS={avg_wait}s, PEAK_HOUR={peak_hour}:00")


class QueueLoadPredictor:
    """Inference engine to predict customer load and wait times at any time & day."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or config.MODELS_DIR
        self.load_model_path = os.path.join(self.models_dir, "lightgbm_load_model.txt")
        self.wait_model_path = os.path.join(self.models_dir, "lightgbm_wait_model.txt")
        self.meta_path = os.path.join(self.models_dir, "load_model_meta.json")

        self.load_booster: Optional[lgb.Booster] = None
        self.wait_booster: Optional[lgb.Booster] = None
        self.meta: Dict[str, Any] = {}

        self.load_models()

    def is_ready(self) -> bool:
        return self.load_booster is not None and self.wait_booster is not None

    def load_models(self) -> bool:
        if os.path.exists(self.load_model_path) and os.path.exists(self.wait_model_path):
            try:
                self.load_booster = lgb.Booster(model_file=self.load_model_path)
                self.wait_booster = lgb.Booster(model_file=self.wait_model_path)
                if os.path.exists(self.meta_path):
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        self.meta = json.load(f)
                return True
            except Exception as e:
                print(f"Warning: Failed to load LightGBM boosters: {e}")
                return False
        return False

    def predict(
        self,
        day: str = "Monday",
        hour: int = 12,
        zone: int = 1,
        known_load: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Predicts:
        - Customer Load (expected count of customers arriving/waiting per hour)
        - Service Wait Duration (seconds per customer)
        - Queue Status (Normal / Busy / ALERT)
        """
        day_clean = str(day).strip().capitalize()
        day_num = DAY_TO_INT.get(day_clean, 0)
        is_weekend = 1 if day_num in [5, 6] else 0
        bin_val = time_of_day_bin(hour)

        sin_hour = math.sin(2 * math.pi * hour / 24.0)
        cos_hour = math.cos(2 * math.pi * hour / 24.0)
        sin_day = math.sin(2 * math.pi * day_num / 7.0)
        cos_day = math.cos(2 * math.pi * day_num / 7.0)

        peak_interaction = 1 if (hour in [8, 9, 12, 13, 17, 18] and is_weekend == 0) else 0

        # 1. Predict Load
        if self.load_booster:
            x_load = np.array([[
                day_num, hour, zone, is_weekend, bin_val,
                sin_hour, cos_hour, sin_day, cos_day, peak_interaction
            ]])
            pred_load = max(0.0, float(self.load_booster.predict(x_load)[0]))
        else:
            # Fallback heuristic
            pred_load = 15.0 if hour in [8, 9, 12, 13] else 6.0

        used_load = known_load if known_load is not None else pred_load

        # 2. Predict Wait Duration (seconds)
        if self.wait_booster:
            x_wait = np.array([[
                day_num, hour, zone, used_load, is_weekend,
                bin_val, sin_hour, cos_hour, sin_day, cos_day, peak_interaction
            ]])
            pred_wait = max(1.0, float(self.wait_booster.predict(x_wait)[0]))
        else:
            pred_wait = float(getattr(config, "AVERAGE_WAIT_TIME_SECONDS", 14.12))

        # Status classification
        if pred_load >= getattr(config, "ALERT_QUEUE_THRESHOLD", 5) * 3:
            status = "ALERT"
        elif pred_load >= getattr(config, "BUSY_QUEUE_THRESHOLD", 3) * 3:
            status = "Busy"
        else:
            status = "Normal"

        m, s = divmod(int(round(pred_wait)), 60)
        wait_formatted = f"{m}m {s}s"

        return {
            "day": day_clean,
            "hour": hour,
            "zone": zone,
            "predicted_customer_load": round(pred_load, 1),
            "predicted_wait_seconds": round(pred_wait, 1),
            "predicted_wait_formatted": wait_formatted,
            "status": status,
            "total_estimated_burden_seconds": round(pred_load * pred_wait, 1),
        }

    def generate_daily_forecast(self, day: str = "Monday", zone: int = 1) -> pd.DataFrame:
        """Generates a full 24-hour load and wait time forecast table for operational planning."""
        rows = []
        for h in range(24):
            res = self.predict(day=day, hour=h, zone=zone)
            rows.append({
                "Hour": f"{h:02d}:00",
                "Predicted Load (Customers)": res["predicted_customer_load"],
                "Expected Wait": res["predicted_wait_formatted"],
                "Wait (Sec)": res["predicted_wait_seconds"],
                "Status": res["status"]
            })
        return pd.DataFrame(rows)
