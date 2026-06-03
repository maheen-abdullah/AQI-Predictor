"""
app.py — Karachi AQI Predictor Dashboard
Run: streamlit run app.py
"""
import json, os, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

CSV_PATH   = Path("data/features.csv")
MODEL_PATH = Path("model/aqi_model.pkl")
META_PATH  = Path("model/model_meta.json")

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")

AQI_LEVELS = [
    (50,  "Good",                    "#00e400", "Air quality is satisfactory."),
    (100, "Moderate",                "#ffff00", "Acceptable; some pollutants may affect sensitive individuals."),
    (150, "Unhealthy for Sensitive", "#ff7e00", "Sensitive groups may experience health effects."),
    (200, "Unhealthy",               "#ff0000", "Everyone may begin to experience health effects."),
    (300, "Very Unhealthy",          "#8f3f97", "Health alert — serious effects for everyone."),
    (500, "Hazardous",               "#7e0023", "Emergency conditions."),
]

def aqi_info(v):
    for threshold, label, color, desc in AQI_LEVELS:
        if v <= threshold:
            return label, color, desc
    return "Hazardous", "#7e0023", "Emergency conditions."

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.title("🌫️ AQI Predictor")
    st.caption("Karachi, Pakistan")
    st.divider()
    st.subheader("AQI Scale")
    for threshold, label, color, _ in AQI_LEVELS:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<div style="width:12px;height:12px;border-radius:3px;background:{color}"></div>'
            f'<span style="font-size:12px">{label} (≤{threshold})</span></div>',
            unsafe_allow_html=True)
    st.divider()
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

st.title("🌫️ Karachi AQI — 3-Day Forecast")

# ── Load model & data ────────────────────────────────────
model, meta, df = None, None, pd.DataFrame()

if MODEL_PATH.exists() and META_PATH.exists():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH).sort_values("timestamp").reset_index(drop=True)

# ── Demo data if nothing trained yet ────────────────────
if df.empty:
    st.info("No data yet. Run `python pipelines/backfill_pipeline.py` then `python pipelines/training_pipeline.py`", icon="ℹ️")
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    df = pd.DataFrame({"timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"), "aqi": np.random.normal(150, 20, 30).clip(80, 220).round(1)})
    for col in ["pm25","pm10","o3","no2","co","so2","temperature","humidity","wind_speed","pressure","dew_point","hour","day_of_week","month","is_weekend","day_of_year"]:
        df[col] = np.random.uniform(10, 100, 30)

# ── Current AQI ──────────────────────────────────────────
current_aqi = float(df["aqi"].iloc[-1])
label, color, desc = aqi_info(current_aqi)

st.markdown(f"""
<div style="background:{color}22;border:2px solid {color};border-radius:12px;
     padding:20px 28px;margin-bottom:20px;display:flex;align-items:center;gap:24px">
  <div style="font-size:64px;font-weight:700;color:{color};line-height:1">{int(current_aqi)}</div>
  <div>
    <div style="font-size:22px;font-weight:600;color:{color}">{label}</div>
    <div style="font-size:13px;opacity:.8;margin-top:4px">{desc}</div>
    <div style="font-size:11px;opacity:.6;margin-top:6px">Current AQI · Karachi</div>
  </div>
</div>""", unsafe_allow_html=True)

if current_aqi > 200:
    st.error("🚨 HAZARDOUS — Avoid all outdoor activities.")
elif current_aqi > 150:
    st.error("⚠️ UNHEALTHY — Sensitive groups should stay indoors.")
elif current_aqi > 100:
    st.warning("⚠️ MODERATE — Consider reducing outdoor activity.")

# ── 3-Day Forecast ───────────────────────────────────────
st.subheader("📅 3-Day Forecast")

FEATURE_COLS = ["pm25","pm10","o3","no2","co","so2","temperature","humidity",
                "wind_speed","pressure","dew_point","hour","day_of_week","month","is_weekend","day_of_year"]

def make_forecast(df, model, days=3):
    forecasts = []
    last = df.iloc[-1].copy()
    for i in range(1, days + 1):
        future = datetime.now(timezone.utc) + timedelta(days=i)
        last["hour"]        = 12
        last["day_of_week"] = future.weekday()
        last["month"]       = future.month
        last["is_weekend"]  = int(future.weekday() >= 5)
        last["day_of_year"] = future.timetuple().tm_yday
        row = pd.DataFrame([last[FEATURE_COLS]])
        if model:
            pred = max(0, round(float(model.predict(row)[0]), 1))
        else:
            pred = round(current_aqi * np.random.uniform(0.9, 1.1), 1)
        l, c, d = aqi_info(pred)
        forecasts.append({"date": future.strftime("%A, %b %d"), "aqi": pred, "label": l, "color": c, "desc": d})
    return forecasts

forecasts = make_forecast(df, model)
cols = st.columns(3)
for col, fc in zip(cols, forecasts):
    with col:
        st.markdown(f"""
        <div style="background:{fc['color']}18;border:1.5px solid {fc['color']}88;
             border-radius:10px;padding:16px;text-align:center">
          <div style="font-size:12px;color:#888;margin-bottom:6px">{fc['date']}</div>
          <div style="font-size:42px;font-weight:700;color:{fc['color']};line-height:1">{int(fc['aqi'])}</div>
          <div style="font-size:13px;font-weight:600;color:{fc['color']};margin-top:4px">{fc['label']}</div>
          <div style="font-size:11px;opacity:.7;margin-top:6px;line-height:1.4">{fc['desc']}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Historical Chart ─────────────────────────────────────
st.subheader("📈 AQI History + Forecast")
hist = df[["timestamp","aqi"]].tail(30).copy()
hist["timestamp"] = pd.to_datetime(hist["timestamp"])

fig = go.Figure()
for lo, hi, c, _ in [(0,50,"rgba(0,228,0,0.05)",""), (50,100,"rgba(255,255,0,0.05)",""),
                      (100,150,"rgba(255,126,0,0.05)",""), (150,200,"rgba(255,0,0,0.05)",""),
                      (200,300,"rgba(143,63,151,0.05)","")]:
    fig.add_hrect(y0=lo, y1=hi, fillcolor=c, line_width=0)

fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["aqi"], mode="lines+markers",
    name="Historical", line=dict(color="#4F8EF7", width=2), marker=dict(size=4)))

future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 4)]
fig.add_trace(go.Scatter(x=future_dates, y=[f["aqi"] for f in forecasts], mode="lines+markers",
    name="Forecast", line=dict(color="#FF6B35", width=2, dash="dash"), marker=dict(size=8, symbol="diamond")))

for val, lbl, c in [(100,"Moderate","#ffff00"),(150,"Unhealthy","#ff7e00"),(200,"Very Unhealthy","#ff0000")]:
    fig.add_hline(y=val, line_dash="dot", line_color=c, line_width=1,
                  annotation_text=lbl, annotation_position="right", opacity=0.5)

fig.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# ── Model info ───────────────────────────────────────────
if meta:
    st.divider()
    st.subheader("🤖 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", meta.get("model_name", "—"))
    c2.metric("RMSE", f"{meta.get('rmse', 0):.2f}")
    c3.metric("MAE",  f"{meta.get('mae',  0):.2f}")
    c4.metric("R²",   f"{meta.get('r2',   0):.3f}")

# ── Pollutants ───────────────────────────────────────────
st.divider()
st.subheader("🔬 Current Pollutant Levels")
pollutants = {"PM2.5":"pm25","PM10":"pm10","O₃":"o3","NO₂":"no2","CO":"co","SO₂":"so2"}
pcols = st.columns(6)
last_row = df.iloc[-1]
for col, (name, key) in zip(pcols, pollutants.items()):
    val = last_row.get(key, np.nan)
    col.metric(name, f"{float(val):.1f}" if pd.notna(val) else "N/A")

st.divider()
st.caption("Karachi AQI Predictor · Data: AQICN · Automated via GitHub Actions")
