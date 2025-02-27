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
    "forecast_horizon": 30,  # Wird jetzt nicht mehr genutzt, sondern nur als Fallback
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
    return pd.DataFrame({
        "ds": pd.date_range(start=start_date, end=end_date),
        "traffic_intensity": np.random.randint(0, 10, size=(end_date - start_date).days + 1)
    })

# Simulierte Eventdaten
def fetch_event_data(start_date, end_date):
    return pd.DataFrame({
        "ds": pd.date_range(start=start_date, end=end_date),
        "event_count": np.random.randint(0, 5, size=(end_date - start_date).days + 1)
    })

# Prophet-Modell trainieren
@st.cache_data
def train_model(data, changepoint_prior_scale, seasonality_prior_scale):
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )
    # Zusätzliche Regressoren hinzufügen, falls vorhanden
    for col in ['temperature', 'humidity', 'traffic_intensity', 'event_count']:
        if col in data.columns:
            model.add_regressor(col)
    model.fit(data)
    return model

st.title("📈 Intelligentes Forecasting Tool")
st.sidebar.header("Einstellungen")

# Daten Upload & Spaltenzuordnung
st.sidebar.subheader("Daten Upload & Spaltenzuordnung")
uploaded_file = st.sidebar.file_uploader("📂 Zeitreihendaten hochladen (CSV)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if data.empty:
        st.sidebar.error("❌ Die hochgeladene Datei ist leer.")
    else:
        # Spalten der hochgeladenen Datei ermitteln
        columns = list(data.columns)
        ds_col = st.sidebar.selectbox("Wähle das Datums-Feld (ds)", columns, key="ds_col")
        y_col = st.sidebar.selectbox("Wähle das Zielvariable-Feld (y)", columns, key="y_col")
        
        # Konvertiere das ausgewählte Datums-Feld in datetime
        data['ds'] = pd.to_datetime(data[ds_col], errors='coerce')
        if data['ds'].isnull().all():
            st.sidebar.error("❌ Das ausgewählte Datums-Feld enthält keine gültigen Datumswerte.")
        else:
            data['y'] = data[y_col]
            st.sidebar.success("✅ Daten erfolgreich geladen!")
            st.write("### Datenvorschau")
            st.write(data.head())
            st.session_state['data'] = data

# Forecast-Horizont: Kein Schieberegler mehr, sondern nur das Enddatum wird angegeben
if 'data' in st.session_state and st.session_state['data'] is not None:
    # Das letzte Datum der historischen Daten ermitteln
    last_date = st.session_state['data']['ds'].max().date()
    st.sidebar.write("Letztes Datum in den historischen Daten:", last_date)
    # Der Nutzer wählt nur das Enddatum des Forecast-Horizonts
    forecast_end = st.sidebar.date_input("Enddatum für Forecast-Horizont", value=last_date + timedelta(days=30))
    # Der Forecast-Horizont wird automatisch als Differenz (in Tagen) berechnet
    forecast_horizon = (forecast_end - last_date).days
else:
    forecast_horizon = DEFAULT_CONFIG["forecast_horizon"]

# Weitere Forecast-Einstellungen
api_key = st.sidebar.text_input("🔑 OpenWeatherMap API Key", type="password")
latitude = st.sidebar.number_input("🌍 Breitengrad", value=52.5200)
longitude = st.sidebar.number_input("🌍 Längengrad", value=13.4050)
changepoint_prior_scale = st.sidebar.slider("🔄 Changepoint Prior Scale", 0.01, 0.5, 0.05)
seasonality_prior_scale = st.sidebar.slider("📊 Seasonality Prior Scale", 0.01, 10.0, 10.0)

# Forecast-Berechnung, falls Daten vorhanden sind
if 'data' in st.session_state and st.session_state['data'] is not None:
    if st.button("🚀 Forecast starten"):
        with st.spinner("📡 Modell wird trainiert..."):
            model = train_model(st.session_state['data'], changepoint_prior_scale, seasonality_prior_scale)
            # Erstelle Future DataFrame anhand des automatisch berechneten Forecast-Horizonts
            future = model.make_future_dataframe(periods=forecast_horizon)
            # Falls zusätzliche Regressoren vorhanden sind, diese hinzufügen (hier simuliert anhand der letzten Werte)
            for col in ['temperature', 'humidity', 'traffic_intensity', 'event_count']:
                if col in st.session_state['data'].columns:
                    if len(st.session_state['data'][col]) >= forecast_horizon:
                        future[col] = st.session_state['data'][col].iloc[-forecast_horizon:].values
                    else:
                        future[col] = np.nan
            forecast = model.predict(future)
            st.session_state['forecast'] = forecast
            st.success("✅ Forecast erfolgreich erstellt!")
            
            # Forecast visualisieren
            st.subheader("📉 Forecast-Visualisierung")
            fig = px.line(forecast, x='ds', y='yhat', title="🔮 Prognose")
            fig.add_scatter(x=st.session_state['data']['ds'], y=st.session_state['data']['y'], mode='lines', name="Tatsächliche Werte")
            st.plotly_chart(fig)
            
            # Performance-Metriken berechnen: Merge der historischen Daten mit Forecast-Daten anhand von 'ds'
            historical = st.session_state['data'][['ds', 'y']]
            merged = pd.merge(historical, forecast[['ds', 'yhat']], on='ds', how='inner')
            if merged.empty:
                st.write("⚠️ Es gibt keine überlappenden Zeitpunkte zwischen den historischen Daten und dem Forecast. Performance-Metriken können nicht berechnet werden.")
            else:
                mae = mean_absolute_error(merged['y'], merged['yhat'])
                mse = mean_squared_error(merged['y'], merged['yhat'])
                r2 = r2_score(merged['y'], merged['yhat'])
                st.write(f"📊 **Metriken:**\n- MAE: {mae:.2f}\n- MSE: {mse:.2f}\n- R²: {r2:.2f}")
            
            # Export-Funktion
            if st.button("💾 Export als CSV"):
                forecast.to_csv("forecast_results.csv", index=False)
                st.success("✅ CSV exportiert!")
