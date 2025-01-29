import os
import json
import pandas as pd
import numpy as np
from prophet import Prophet
import requests
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
from joblib import Parallel, delayed
from dotenv import load_dotenv

# Standard-Konfigurationsdatei
CONFIG_FILE = "config.json"

# Standardkonfiguration
DEFAULT_CONFIG = {
    "forecast_horizon": 30,
    "api_keys": {
        "openweathermap": "Ihr_API_Schlüssel"
    },
    "location": {
        "latitude": 52.5200,
        "longitude": 13.4050
    }
}

# Konfigurationsdatei laden
def load_config():
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

# Konfigurationsdatei speichern
def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

# Wetterdaten abrufen
def fetch_weather_data(start_date, end_date, lat, lon, api_key):
    weather_data = []
    current_date = start_date
    while current_date <= end_date:
        timestamp = int(current_date.timestamp())
        url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
        try:
            response = requests.get(url, params={
                "lat": lat,
                "lon": lon,
                "dt": timestamp,
                "appid": api_key,
                "units": "metric"
            })
            response.raise_for_status()  # Wirft eine Ausnahme bei HTTP-Fehlern
            data = response.json()
            weather_data.append({
                "ds": current_date,
                "temperature": data['current']['temp'],
                "humidity": data['current']['humidity']
            })
        except requests.exceptions.RequestException as e:
            st.error(f"Fehler beim Abrufen der Wetterdaten: {e}")
            break
        current_date += timedelta(days=1)
    return pd.DataFrame(weather_data)

# Verkehrsdaten (simuliert)
def fetch_traffic_data(start_date, end_date):
    traffic_data = []
    for single_date in pd.date_range(start=start_date, end=end_date):
        traffic_intensity = np.random.randint(0, 10)
        traffic_data.append({
            "ds": single_date,
            "traffic_intensity": traffic_intensity
        })
    return pd.DataFrame(traffic_data)

# Eventdaten (simuliert)
def fetch_event_data(start_date, end_date):
    event_data = []
    for single_date in pd.date_range(start=start_date, end=end_date):
        event_count = np.random.randint(0, 5)
        event_data.append({
            "ds": single_date,
            "event_count": event_count
        })
    return pd.DataFrame(event_data)

# Modelltraining (mit Caching für bessere Performance)
@st.cache(allow_output_mutation=True)
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
st.title("Forecasting Tool")
st.sidebar.header("Einstellungen")

# Anleitung
with st.expander("Wie verwende ich dieses Tool?"):
    st.write("""
    1. Laden Sie eine CSV-Datei mit Zeitreihendaten hoch (Spalten: `ds` für Datum, `y` für Zielvariable).
    2. Fügen Sie optional Meta-Daten hinzu (Wetter, Verkehr, Events).
    3. Passen Sie die Modellparameter an.
    4. Starten Sie den Forecast und visualisieren Sie die Ergebnisse.
    5. Exportieren Sie die Ergebnisse in verschiedenen Formaten.
    """)

# Einstellungen bearbeiten
forecast_horizon = st.sidebar.number_input("Forecast-Horizont (Tage)", min_value=1, max_value=365, value=30)
api_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")
latitude = st.sidebar.number_input("Breitengrad", value=52.5200)
longitude = st.sidebar.number_input("Längengrad", value=13.4050)
changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05)
seasonality_prior_scale = st.sidebar.slider("Seasonality Prior Scale", 0.01, 10.0, 10.0)

# Datei hochladen
uploaded_file = st.file_uploader("Zeitreihendaten hochladen (CSV-Datei)", type="csv")
data = None
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if 'ds' not in data.columns or 'y' not in data.columns:
        st.error("Die CSV-Datei muss die Spalten 'ds' (Datum) und 'y' (Zielvariable) enthalten.")
    else:
        data['ds'] = pd.to_datetime(data['ds'])
        st.write("Daten erfolgreich geladen:", data.head())

# Meta-Daten hinzufügen
if data is not None and st.button("Meta-Daten hinzufügen"):
    try:
        start_date = data['ds'].min()
        end_date = data['ds'].max()
        weather = fetch_weather_data(start_date, end_date, latitude, longitude, api_key)
        traffic = fetch_traffic_data(start_date, end_date)
        events = fetch_event_data(start_date, end_date)
        meta_data = weather.merge(traffic, on="ds", how="outer").merge(events, on="ds", how="outer")
        
        # Überprüfen, ob Spalten bereits vorhanden sind
        for col in meta_data.columns:
            if col in data.columns and col != 'ds':
                st.warning(f"Spalte '{col}' existiert bereits und wird überschrieben.")
        
        data = data.merge(meta_data, on="ds", how="left")
        st.success("Meta-Daten erfolgreich hinzugefügt!")
        st.write(data.head())
    except Exception as e:
        st.error(f"Fehler beim Hinzufügen von Meta-Daten: {e}")

# Forecast starten
if data is not None and st.button("Forecast starten"):
    try:
        with st.spinner("Modell wird trainiert..."):
            model = train_model(data, changepoint_prior_scale, seasonality_prior_scale)
            future = model.make_future_dataframe(periods=forecast_horizon)
            for col in ['temperature', 'humidity', 'traffic_intensity', 'event_count']:
                if col in data.columns:
                    future[col] = data[col].iloc[-forecast_horizon:].values
            forecast = model.predict(future)
            st.session_state['forecast'] = forecast
            st.session_state['model'] = model
            st.session_state['data'] = data
        st.success("Forecast erfolgreich erstellt!")

        # Plot Forecast mit Plotly
        st.subheader("Forecast-Visualisierung")
        fig = px.line(forecast, x='ds', y='yhat', title="Forecast mit Meta-Daten")
        if 'y' in data.columns:
            fig.add_scatter(x=data['ds'], y=data['y'], mode='lines', name="Tatsächliche Werte")
        st.plotly_chart(fig)

        # Zusätzliche Komponenten visualisieren
        st.subheader("Forecast-Komponenten")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # Metriken anzeigen (falls echte Werte vorhanden sind)
        if 'y' in data.columns:
            mae = mean_absolute_error(data['y'], forecast['yhat'][:len(data)])
            mse = mean_squared_error(data['y'], forecast['yhat'][:len(data)])
            r2 = r2_score(data['y'], forecast['yhat'][:len(data)])
            st.write(f"**Metriken:**")
            st.write(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

    except Exception as e:
        st.error(f"Fehler beim Forecast: {e}")

# Ergebnisse exportieren
if "forecast" in st.session_state:
    forecast = st.session_state["forecast"]
    save_path = st.text_input("Pfad für die gespeicherte Datei", "forecast_results.csv")
    if st.button("Ergebnisse als CSV exportieren"):
        forecast.to_csv(save_path, index=False)
        st.success("Ergebnisse erfolgreich exportiert!")
    if st.button("Ergebnisse als Excel exportieren"):
        forecast.to_excel(save_path.replace(".csv", ".xlsx"), index=False)
        st.success("Ergebnisse erfolgreich exportiert!")
else:
    st.warning("Bitte führen Sie zuerst den Forecast durch.")
