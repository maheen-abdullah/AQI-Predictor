# 🌫️ Karachi AQI Predictor

Predicts Air Quality Index for Karachi for the next 3 days using Machine Learning.

## How it works
- **Feature pipeline** fetches live AQI data from AQICN every hour → saves to `data/features.csv`
- **Training pipeline** trains 4 ML models daily → saves best to `model/aqi_model.pkl`
- **Dashboard** loads the model and shows live + 3-day forecast

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/aqi-predictor.git
cd aqi-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# 3. Install libraries
pip install -r requirements.txt

# 4. Add your AQICN key
cp .env.example .env
# Open .env and paste your AQICN_API_KEY

# 5. Load historical data
python pipelines/backfill_pipeline.py

# 6. Train the model
python pipelines/training_pipeline.py

# 7. Launch dashboard
streamlit run app.py
```

## Deploy Free on Streamlit Cloud
1. Push to GitHub
2. Go to share.streamlit.io → New app → select your repo
3. Add secret: `AQICN_API_KEY = "your_key"`
4. Deploy

## GitHub Actions Automation
Add `AQICN_API_KEY` in GitHub → Settings → Secrets → Actions.
- Feature pipeline runs every hour
- Training pipeline runs every day at 2 AM

## Tech Stack
| Layer | Tool |
|-------|------|
| Data | AQICN API |
| Storage | CSV files (local + GitHub) |
| ML | scikit-learn, XGBoost |
| Dashboard | Streamlit + Plotly |
| Automation | GitHub Actions |
