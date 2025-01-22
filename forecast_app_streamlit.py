{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww24360\viewh15360\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import json\
import pandas as pd\
import numpy as np\
from prophet import Prophet\
import requests\
import streamlit as st\
from datetime import datetime, timedelta\
\
# Standard-Konfigurationsdatei\
CONFIG_FILE = "config.json"\
\
# Standardkonfiguration\
DEFAULT_CONFIG = \{\
    "forecast_horizon": 30,\
    "api_keys": \{\
        "openweathermap": "Ihr_API_Schl\'fcssel"\
    \},\
    "location": \{\
        "latitude": 52.5200,\
        "longitude": 13.4050\
    \}\
\}\
\
# Konfigurationsdatei laden\
def load_config():\
    if not os.path.exists(CONFIG_FILE):\
        save_config(DEFAULT_CONFIG)\
    with open(CONFIG_FILE, "r") as f:\
        return json.load(f)\
\
# Konfigurationsdatei speichern\
def save_config(config):\
    with open(CONFIG_FILE, "w") as f:\
        json.dump(config, f, indent=4)\
\
# Wetterdaten abrufen\
def fetch_weather_data(start_date, end_date, lat, lon, api_key):\
    weather_data = []\
    current_date = start_date\
    while current_date <= end_date:\
        timestamp = int(current_date.timestamp())\
        url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"\
        response = requests.get(url, params=\{\
            "lat": lat,\
            "lon": lon,\
            "dt": timestamp,\
            "appid": api_key,\
            "units": "metric"\
        \})\
        if response.status_code == 200:\
            data = response.json()\
            weather_data.append(\{\
                "ds": current_date,\
                "temperature": data['current']['temp'],\
                "humidity": data['current']['humidity']\
            \})\
        current_date += timedelta(days=1)\
    return pd.DataFrame(weather_data)\
\
# Verkehrsdaten (simuliert)\
def fetch_traffic_data(start_date, end_date):\
    traffic_data = []\
    for single_date in pd.date_range(start=start_date, end=end_date):\
        traffic_intensity = np.random.randint(0, 10)\
        traffic_data.append(\{\
            "ds": single_date,\
            "traffic_intensity": traffic_intensity\
        \})\
    return pd.DataFrame(traffic_data)\
\
# Eventdaten (simuliert)\
def fetch_event_data(start_date, end_date):\
    event_data = []\
    for single_date in pd.date_range(start=start_date, end=end_date):\
        event_count = np.random.randint(0, 5)\
        event_data.append(\{\
            "ds": single_date,\
            "event_count": event_count\
        \})\
    return pd.DataFrame(event_data)\
\
# Konfiguration laden\
config = load_config()\
\
# Streamlit UI\
st.title("Forecasting Tool")\
st.sidebar.header("Einstellungen")\
\
# Einstellungen bearbeiten\
forecast_horizon = st.sidebar.number_input("Forecast-Horizont (Tage)", min_value=1, value=config["forecast_horizon"])\
api_key = st.sidebar.text_input("OpenWeatherMap API Key", config["api_keys"]["openweathermap"], type="password")\
latitude = st.sidebar.number_input("Breitengrad", value=config["location"]["latitude"])\
longitude = st.sidebar.number_input("L\'e4ngengrad", value=config["location"]["longitude"])\
\
# Speichern der Einstellungen\
if st.sidebar.button("Einstellungen speichern"):\
    config["forecast_horizon"] = forecast_horizon\
    config["api_keys"]["openweathermap"] = api_key\
    config["location"] = \{"latitude": latitude, "longitude": longitude\}\
    save_config(config)\
    st.sidebar.success("Einstellungen gespeichert!")\
\
# Datei hochladen\
uploaded_file = st.file_uploader("Zeitreihendaten hochladen (CSV-Datei)", type="csv")\
data = None\
if uploaded_file:\
    data = pd.read_csv(uploaded_file)\
    data['ds'] = pd.to_datetime(data['ds'])\
    st.write("Daten erfolgreich geladen:", data.head())\
\
# Meta-Daten hinzuf\'fcgen\
if data is not None and st.button("Meta-Daten hinzuf\'fcgen"):\
    try:\
        start_date = data['ds'].min()\
        end_date = data['ds'].max()\
        weather = fetch_weather_data(start_date, end_date, latitude, longitude, api_key)\
        traffic = fetch_traffic_data(start_date, end_date)\
        events = fetch_event_data(start_date, end_date)\
        meta_data = weather.merge(traffic, on="ds", how="outer").merge(events, on="ds", how="outer")\
        data = data.merge(meta_data, on="ds", how="left")\
        st.success("Meta-Daten erfolgreich hinzugef\'fcgt!")\
        st.write(data.head())\
    except Exception as e:\
        st.error(f"Fehler beim Hinzuf\'fcgen von Meta-Daten: \{e\}")\
\
# Forecast starten\
if data is not None and st.button("Forecast starten"):\
    try:\
        model = Prophet()\
        for col in ['temperature', 'humidity', 'traffic_intensity', 'event_count']:\
            if col in data.columns:\
                model.add_regressor(col)\
        model.fit(data)\
        future = model.make_future_dataframe(periods=forecast_horizon)\
        for col in ['temperature', 'humidity', 'traffic_intensity', 'event_count']:\
            if col in data.columns:\
                future[col] = data[col].iloc[-forecast_horizon:].values\
        forecast = model.predict(future)\
\
        st.success("Forecast erfolgreich erstellt!")\
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())\
\
        # Plot anzeigen\
        st.line_chart(\{\
            "Vorhersage": forecast.set_index("ds")["yhat"],\
            "Unsicherheitsintervall (obere Grenze)": forecast.set_index("ds")["yhat_upper"],\
            "Unsicherheitsintervall (untere Grenze)": forecast.set_index("ds")["yhat_lower"]\
        \})\
    except Exception as e:\
        st.error(f"Fehler beim Forecast: \{e\}")\
\
# Ergebnisse speichern\
if data is not None and st.button("Ergebnisse speichern"):\
    save_path = st.text_input("Pfad f\'fcr die gespeicherte Datei", "forecast_results.csv")\
    if save_path:\
        forecast.to_csv(save_path, index=False)\
        st.success("Ergebnisse erfolgreich gespeichert!")\
}