import os
import json
import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import datetime, timedelta
from prophet import Prophet
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed

# Standard-Konfigurationsdatei
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "forecast_horizon": 30,
    "api_keys": {"openweathermap": "Ihr_API_Schlüssel"},
    "location": {"latitude": 52.5200, "longitude": 13.4050}
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

# Wetterdaten effizient abrufen
def fetch_weather_data(start_date, end_date, lat, lon, api_key):
    date_range = pd.date_range(start=start_date, end=end_date)
    weather_data = []
    
    def fetch_single_day(date):
        url = "http://api.openweathermap.org/data/2.5/onecall/timemachine"
        timestamp = int(date.timestamp())
        try:
            response = requests.get(url, params={"lat": lat, "lon": lon, "dt": timestamp, "appid": api_key, "units": "metric"})
            response.raise_for_status()
            data = response.json()
            return {"ds": date, "temperature": data['current']['temp'], "humidity": data['current']['humidity']}
        except requests.exceptions.RequestException:
            return {"ds": date, "temperature": np.nan, "humidity": np.nan}
    
    weather_data = Parallel(n_jobs=-1)(delayed(fetch_single_day)(date) for date in date_range)
    return pd.DataFrame(weather_data)

# Simulierte Verkehrsdaten
def fetch_traffic_data(start_date, end_date):
    return pd.DataFrame({"ds": pd.date_range(start=start_date, end=end_date), "traffic_intensity": np.random.randint(0, 10, size=(end_date - start_date).days + 1)})

# Simulierte Eventdaten
def fetch_event_data(start_date, end_date):
    return pd.DataFrame({"ds": pd.date_range(start=start_date, end=end_date), "event_count": np.random.randint(0, 5, size=(end_date - start_date).days + 1)})

# Prophet-Modell trainieren
@st.cache_data
def train_model(data, changepoint_prior_scale, seasonality_prior_scale):
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    for col in ['temperature', 'humidity', 'traffic_intensity', 'event_count']:
        if col in data.columns:
            model.add_regressor(col)
    model.fit(data)
    return model

# Streamlit UI
st.title("📈 Intelligentes Forecasting Tool")
st.sidebar.header("Einstellungen")

forecast_horizon = st.sidebar.slider("Forecast-Horizont (Tage)", 1, 365, 30)
api_key = st.sidebar.text_input("🔑 OpenWeatherMap API Key", type="password")
latitude = st.sidebar.number_input("🌍 Breitengrad", value=52.5200)
longitude = st.sidebar.number_input("🌍 Längengrad", value=13.4050)
changepoint_prior_scale = st.sidebar.slider("🔄 Changepoint Prior Scale", 0.01, 0.5, 0.05)
seasonality_prior_scale = st.sidebar.slider("📊 Seasonality Prior Scale", 0.01, 10.0, 10.0)

# Datei hochladen
uploaded_file = st.file_uploader("📂 Zeitreihendaten hochladen (CSV)", type="csv")
data = None
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if 'ds' not in data.columns or 'y' not in data.columns:
        st.error("❌ Die Datei muss 'ds' (Datum) und 'y' (Zielvariable) enthalten.")
    else:
        data['ds'] = pd.to_datetime(data['ds'])
        st.write("✅ Daten geladen:", data.head())

# requirements.txt
dependencies = """
pandas
numpy
requests
streamlit
prophet
plotly
scikit-learn
joblib
pystan==2.19.1.1
cmdstanpy
"""
with open("requirements.txt", "w") as f:
    f.write(dependencies)
