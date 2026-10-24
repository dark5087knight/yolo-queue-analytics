"""
Train Script for LightGBM Queue Load & Wait Time Prediction.

Trains LightGBM gradient boosting regressors on extracted customer queue data
(Customer ID, Waiting Time, Hour, Day, Zone, Entry Time) to accurately predict:
  1. Hourly Customer Load (traffic intensity and arrivals)
  2. Load Time (service waiting duration in seconds)
  3. Dynamic Peak Operational Hours

Usage:
  python train.py
  python train.py --generate-synthetic
  python train.py --data reports/sample_queue_data.csv
  python train.py --predict --day Monday --hour 13
  python train.py --forecast Monday
"""

import os
import sys
import argparse
from typing import Optional

# Ensure project directory is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
from load_predictor import (
    QueueDataProcessor,
    LightGBMQueueTrainer,
    QueueLoadPredictor,
    generate_synthetic_historical_data,
    DAY_NAMES,
)


def find_default_csv() -> str:
    """Finds the best available queue data CSV in reports directory."""
    synthetic_path = os.path.join(config.REPORTS_DIR, "historical_queue_dataset.csv")
    sample_path = os.path.join(config.REPORTS_DIR, "sample_queue_data.csv")

    if os.path.exists(synthetic_path):
        return synthetic_path
    if os.path.exists(sample_path):
        return sample_path

    # Check for any .csv in reports dir
    if os.path.exists(config.REPORTS_DIR):
        csv_files = [f for f in os.listdir(config.REPORTS_DIR) if f.endswith(".csv")]
        if csv_files:
            return os.path.join(config.REPORTS_DIR, csv_files[0])

    return sample_path


def run_training(
    csv_path: Optional[str] = None,
    generate_synthetic: bool = False,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    update_config: bool = True,
):
    print("=" * 70)
    print("   LightGBM Queue Load & Wait Time Model Training Pipeline   ")
    print("=" * 70)

    # 1. Determine or generate input dataset
    if csv_path is None or not os.path.exists(csv_path):
        csv_path = find_default_csv()

    if generate_synthetic or not os.path.exists(csv_path):
        synthetic_path = os.path.join(config.REPORTS_DIR, "historical_queue_dataset.csv")
        print(f"\n[INFO] Generating realistic historical queue dataset (4 weeks of logs)...")
        generate_synthetic_historical_data(synthetic_path, days=28)
        csv_path = synthetic_path

    print(f"\n[1/4] Loading extracted queue dataset: {csv_path}")
    processor = QueueDataProcessor(csv_path)
    df_raw = processor.load_and_clean()
    print(f"      • Loaded {len(df_raw)} customer records.")

    # If dataset has very few rows (e.g. sample_queue_data with 15 rows), offer augmentation
    if len(df_raw) < 50:
        print(f"\n[!] Notice: Dataset has only {len(df_raw)} rows.")
        print(f"    Generating comprehensive 4-week historical dataset for robust gradient boosting...")
        synthetic_path = os.path.join(config.REPORTS_DIR, "historical_queue_dataset.csv")
        generate_synthetic_historical_data(synthetic_path, days=28)
        processor = QueueDataProcessor(synthetic_path)
        df_raw = processor.load_and_clean()
        print(f"      • Expanded dataset: {len(df_raw)} records across all weekdays and hours.")

    # 2. Build Feature Datasets
    print("\n[2/4] Engineering Temporal & Queue Features...")
    df_hourly, df_individual = processor.build_feature_datasets()
    print(f"      • Hourly Traffic Slots:   {len(df_hourly)} samples")
    print(f"      • Customer Transactions:  {len(df_individual)} samples")

    # 3. Train LightGBM Regressors
    print("\n[3/4] Training LightGBM Gradient Boosted Decision Trees...")
    trainer = LightGBMQueueTrainer(models_dir=config.MODELS_DIR)
    results = trainer.train_models(
        df_hourly=df_hourly,
        df_individual=df_individual,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
    )

    # 4. Synchronize config.py
    if update_config:
        trainer.update_config(
            avg_wait=results["calib_avg_wait"],
            peak_hour=results["peak_hour"],
        )

    # 5. Verification Inference
    print("\n[4/4] Verifying Inference Engine with Sample Predictions:")
    predictor = QueueLoadPredictor(models_dir=config.MODELS_DIR)

    test_queries = [
        ("Monday", 8),    # Morning rush
        ("Monday", 13),   # Lunch peak
        ("Tuesday", 15),  # Normal afternoon
        ("Friday", 17),   # Friday evening rush
        ("Sunday", 12),   # Weekend lunch
    ]

    print("\n" + "-" * 75)
    print(f"{'Day':<10} | {'Hour':<6} | {'Predicted Load (Cust/hr)':<26} | {'Wait Time':<12} | {'Status':<8}")
    print("-" * 75)
    for day, hr in test_queries:
        pred = predictor.predict(day=day, hour=hr)
        print(
            f"{pred['day']:<10} | {hr:02d}:00  | "
            f"{pred['predicted_customer_load']:<26.1f} | "
            f"{pred['predicted_wait_formatted']:<12} | "
            f"{pred['status']:<8}"
        )
    print("-" * 75)
    print("\n[SUCCESS] LightGBM queue load & wait time model is ready for production!\n")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Queue Load and Wait Time Predictor")
    parser.add_argument("--data", type=str, default=None, help="Path to exported queue CSV")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate rich historical training dataset")
    parser.add_argument("--n-estimators", type=int, default=250, help="Number of boosting iterations")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--predict", action="store_true", help="Run a test prediction")
    parser.add_argument("--day", type=str, default="Monday", help="Day of week for prediction")
    parser.add_argument("--hour", type=int, default=12, help="Hour of day (0-23) for prediction")
    parser.add_argument("--forecast", type=str, default=None, help="Print 24-hour forecast table for a day (e.g. Monday)")
    parser.add_argument("--no-config-update", action="store_true", help="Do not update config.py")

    args = parser.parse_args()

    if args.forecast:
        predictor = QueueLoadPredictor()
        if not predictor.is_ready():
            print("Model not found. Running training first...")
            run_training()
            predictor = QueueLoadPredictor()

        df_fc = predictor.generate_daily_forecast(day=args.forecast)
        print(f"\n24-Hour Operational Load Forecast for {args.forecast}:")
        print(df_fc.to_string(index=False))
        return

    if args.predict:
        predictor = QueueLoadPredictor()
        if not predictor.is_ready():
            print("Model not found. Running training first...")
            run_training()
            predictor = QueueLoadPredictor()

        pred = predictor.predict(day=args.day, hour=args.hour)
        print("\n" + "=" * 50)
        print(f" Prediction for {pred['day']} at {pred['hour']:02d}:00")
        print("=" * 50)
        print(f" • Predicted Customer Load:   {pred['predicted_customer_load']} customers/hour")
        print(f" • Predicted Wait Duration:   {pred['predicted_wait_seconds']}s ({pred['predicted_wait_formatted']})")
        print(f" • Predicted Queue Status:     {pred['status']}")
        print(f" • Estimated Hourly Burden:   {pred['total_estimated_burden_seconds']:.1f} total service seconds")
        print("=" * 50)
        return

    run_training(
        csv_path=args.data,
        generate_synthetic=args.generate_synthetic,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        update_config=not args.no_config_update,
    )


if __name__ == "__main__":
    main()
