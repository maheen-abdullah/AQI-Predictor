"""
feature_pipeline.py
Fetches the latest AQI reading and appends it to data/features.csv
Run manually or via GitHub Actions every hour.
"""
import os, sys, logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
AQICN_TOKEN = os.getenv("AQICN_API_KEY")
CITY        = os.getenv("CITY", "karachi")
DATA_DIR    = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH    = DATA_DIR / "features.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def fetch_and_build():
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    d = r.json()
    if d.get("status") != "ok":
        raise ValueError(f"API error: {d}")
    raw  = d["data"]
    iaqi = raw.get("iaqi", {})
    def g(k, default): return float(iaqi.get(k, {}).get("v", default))

    ts_str = raw.get("time", {}).get("iso", datetime.now(timezone.utc).isoformat())
    ts = pd.to_datetime(ts_str, utc=True)
    aqi = float(raw.get("aqi", np.nan))

    cat = ("good" if aqi <= 50 else "moderate" if aqi <= 100 else
           "unhealthy_sensitive" if aqi <= 150 else "unhealthy" if aqi <= 200 else
           "very_unhealthy" if aqi <= 300 else "hazardous")

    return {
        "timestamp":   ts.strftime("%Y-%m-%d %H:%M:%S"),
        "city":        CITY,
        "aqi":         aqi,
        "pm25":        g("pm25", np.nan),
        "pm10":        g("pm10", np.nan),
        "o3":          g("o3",   np.nan),
        "no2":         g("no2",  np.nan),
        "co":          g("co",   np.nan),
        "so2":         g("so2",  np.nan),
        "temperature": g("t",    np.nan),
        "humidity":    g("h",    np.nan),
        "wind_speed":  g("w",    np.nan),
        "pressure":    g("p",    np.nan),
        "dew_point":   g("d",    np.nan),
        "hour":        ts.hour,
        "day_of_week": ts.dayofweek,
        "month":       ts.month,
        "is_weekend":  int(ts.dayofweek >= 5),
        "day_of_year": ts.dayofyear,
        "aqi_category": cat,
    }


def main():
    if not AQICN_TOKEN or AQICN_TOKEN == "your_aqicn_token_here":
        log.error("Set AQICN_API_KEY in your .env file")
        sys.exit(1)

    row = fetch_and_build()
    df_new = pd.DataFrame([row])
    log.info(f"Fetched AQI={row['aqi']} for {CITY} at {row['timestamp']}")

    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, df_new]).drop_duplicates(subset=["timestamp", "city"]).reset_index(drop=True)
    else:
        df = df_new

    df.to_csv(CSV_PATH, index=False)
    log.info(f"✅ Saved. Total rows in CSV: {len(df)}")


if __name__ == "__main__":
    main()
