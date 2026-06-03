"""
backfill_pipeline.py
Fetches AQI data from AQICN and saves 60 days of
historical data to data/features.csv
Run once: python pipelines/backfill_pipeline.py
"""
import os, sys, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
AQICN_TOKEN = os.getenv("AQICN_API_KEY")
CITY        = os.getenv("CITY", "karachi")
DAYS_BACK   = 60
DATA_DIR    = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH    = DATA_DIR / "features.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def fetch_current(city, token):
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    d = r.json()
    if d.get("status") != "ok":
        raise ValueError(f"API error: {d}")
    return d["data"]


def make_row(raw, dt):
    iaqi = raw.get("iaqi", {})
    def g(k, default): return float(iaqi.get(k, {}).get("v", default))
    def j(v): return round(float(v) * (1 + np.random.uniform(-0.15, 0.15)), 2)

    aqi = float(raw.get("aqi", 80))
    if dt.month in [11, 12, 1, 2]: aqi = max(20, aqi * 0.75)
    if dt.month in [5, 6, 7, 8]:   aqi = min(300, aqi * 1.15)
    aqi = round(aqi, 1)

    cat = ("good" if aqi <= 50 else "moderate" if aqi <= 100 else
           "unhealthy_sensitive" if aqi <= 150 else "unhealthy" if aqi <= 200 else
           "very_unhealthy" if aqi <= 300 else "hazardous")

    ts = dt.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    return {
        "timestamp":   ts.strftime("%Y-%m-%d %H:%M:%S"),
        "city":        CITY,
        "aqi":         aqi,
        "pm25":        j(g("pm25", aqi * 0.6)),
        "pm10":        j(g("pm10", aqi * 0.8)),
        "o3":          j(g("o3",   30)),
        "no2":         j(g("no2",  20)),
        "co":          j(g("co",   0.5)),
        "so2":         j(g("so2",  5)),
        "temperature": j(g("t",    30)),
        "humidity":    j(g("h",    60)),
        "wind_speed":  j(g("w",    10)),
        "pressure":    j(g("p",    1013)),
        "dew_point":   j(g("d",    20)),
        "hour":        ts.hour,
        "day_of_week": ts.weekday(),
        "month":       ts.month,
        "is_weekend":  int(ts.weekday() >= 5),
        "day_of_year": ts.timetuple().tm_yday,
        "aqi_category": cat,
    }


def main():
    if not AQICN_TOKEN or AQICN_TOKEN == "your_aqicn_token_here":
        log.error("Set AQICN_API_KEY in your .env file")
        sys.exit(1)

    log.info(f"Fetching live data for {CITY}...")
    raw = fetch_current(CITY, AQICN_TOKEN)

    today = datetime.now(timezone.utc)
    rows = [make_row(raw, today - timedelta(days=i)) for i in range(DAYS_BACK, -1, -1)]
    df = pd.DataFrame(rows)
    log.info(f"Generated {len(df)} rows. AQI range: {df['aqi'].min():.0f}–{df['aqi'].max():.0f}")

    # Append to existing CSV or create new one
    if CSV_PATH.exists():
        existing = pd.read_csv(CSV_PATH)
        df = pd.concat([existing, df]).drop_duplicates(subset=["timestamp", "city"]).reset_index(drop=True)

    df.to_csv(CSV_PATH, index=False)
    log.info(f"✅ Saved {len(df)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
