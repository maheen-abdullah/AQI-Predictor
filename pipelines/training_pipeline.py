"""
training_pipeline.py
Reads data/features.csv, trains ML models, saves best to model/aqi_model.pkl
Run: python pipelines/training_pipeline.py
"""
import os, sys, json, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

CSV_PATH   = Path("data/features.csv")
MODEL_DIR  = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "aqi_model.pkl"
META_PATH  = MODEL_DIR / "model_meta.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2", "co", "so2",
    "temperature", "humidity", "wind_speed", "pressure", "dew_point",
    "hour", "day_of_week", "month", "is_weekend", "day_of_year",
]
TARGET = "aqi"


def load_data():
    if not CSV_PATH.exists():
        log.error(f"{CSV_PATH} not found. Run backfill_pipeline.py first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH).sort_values("timestamp").reset_index(drop=True)
    log.info(f"Loaded {len(df)} rows from {CSV_PATH}")
    return df


def prepare(df):
    df = df.dropna(subset=[TARGET])
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0
    return df[FEATURE_COLS], df[TARGET]


def train_all(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest":      RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost":           xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0),
        "GradientBoosting":  GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        "Ridge":             Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10))]),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2}
        log.info(f"  {name:<22} RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return results


def main():
    df = load_data()
    X, y = prepare(df)

    if len(X) < 10:
        log.error("Not enough data. Run backfill_pipeline.py first.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log.info(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

    results = train_all(X_train, X_test, y_train, y_test)

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best = results[best_name]
    log.info(f"\n🏆 Best model: {best_name} — RMSE={best['rmse']:.2f}, R²={best['r2']:.3f}")

    joblib.dump(best["model"], MODEL_PATH)
    meta = {
        "model_name":   best_name,
        "feature_cols": FEATURE_COLS,
        "rmse": round(best["rmse"], 4),
        "mae":  round(best["mae"],  4),
        "r2":   round(best["r2"],   4),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"✅ Model saved to {MODEL_PATH}")
    log.info(f"✅ Metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
