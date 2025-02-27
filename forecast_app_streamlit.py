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
import matplotlib.pyplot as plt  # Für die Komponenten-Plots

# Zusätzliche Bibliotheken für Google Sheets (müssen in requirements.txt stehen)
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Standard-Konfigurationsdatei
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "forecast_horizon": 30,  # Fallback-Wert (wird nicht mehr aktiv genutzt, wenn Daten vorliegen)
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

# Funktion zum Abrufen von Wetterdaten (Beispiel)
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
    for col in ['temperature', 'humidity', 'traffic_intensity', 'event_count']:
        if col in data.columns:
            model.add_regressor(col)
    model.fit(data)
    return model

st.title("📈 Intelligentes Forecasting Tool")
st.sidebar.header("Einstellungen")

# ─────────────────────────────
# Bereich: Datenquelle auswählen: Datei oder Google Sheets
# ─────────────────────────────
data_source = st.sidebar.radio("Datenquelle", options=["Datei (CSV/Excel)", "Google Sheets"])

if data_source == "Datei (CSV/Excel)":
    uploaded_file = st.sidebar.file_uploader("📂 Datei hochladen (CSV oder Excel)", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        # Dateityp prüfen
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(uploaded_file)
        else:
            st.sidebar.error("Nicht unterstützter Dateityp.")
        
        if data.empty:
            st.sidebar.error("❌ Die hochgeladene Datei ist leer.")
        else:
            columns = list(data.columns)
            ds_col = st.sidebar.selectbox("Wähle das Datums-Feld (ds)", columns, key="ds_col")
            y_col = st.sidebar.selectbox("Wähle das Zielvariable-Feld (y)", columns, key="y_col")
            # Hier erfolgt die Konvertierung der ds-Spalte in datetime
            data['ds'] = pd.to_datetime(data[ds_col], errors='coerce')
            if data['ds'].isnull().all():
                st.sidebar.error("❌ Das ausgewählte Datums-Feld enthält keine gültigen Datumswerte.")
            else:
                data['y'] = data[y_col]
                st.sidebar.success("✅ Daten erfolgreich geladen!")
                st.write("### Datenvorschau")
                st.write(data.head())
                st.session_state['data'] = data

elif data_source == "Google Sheets":
    sheet_url = st.sidebar.text_input("Google Sheets URL", value="")
    cred_path = st.sidebar.text_input("Pfad zur Credentials JSON", value="credentials.json")
    if sheet_url:
        try:
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(cred_path, scope)
            client = gspread.authorize(creds)
            sheet = client.open_by_url(sheet_url)
            worksheet = sheet.get_worksheet(0)  # Verwende das erste Arbeitsblatt
            data = pd.DataFrame(worksheet.get_all_records())
            if data.empty:
                st.sidebar.error("❌ Das Google Sheet ist leer.")
            else:
                columns = list(data.columns)
                ds_col = st.sidebar.selectbox("Wähle das Datums-Feld (ds)", columns, key="ds_col")
                y_col = st.sidebar.selectbox("Wähle das Zielvariable-Feld (y)", columns, key="y_col")
                data['ds'] = pd.to_datetime(data[ds_col], errors='coerce')
                if data['ds'].isnull().all():
                    st.sidebar.error("❌ Das ausgewählte Datums-Feld enthält keine gültigen Datumswerte.")
                else:
                    data['y'] = data[y_col]
                    st.sidebar.success("✅ Daten erfolgreich geladen!")
                    st.write("### Datenvorschau")
                    st.write(data.head())
                    st.session_state['data'] = data
        except Exception as e:
            st.error(f"Fehler beim Laden der Google Sheets Daten: {e}")
    else:
        st.info("Bitte geben Sie die Google Sheets URL ein.")

# ─────────────────────────────
# Bereich: Manuelle Monatsvolumen (Einträge mit Monat, Jahr, Volumen und Checkbox)
# ─────────────────────────────
st.sidebar.subheader("Manuelle Monatsvolumen")
with st.sidebar.form("manual_volumes_form", clear_on_submit=True):
    month = st.selectbox("Monat", options=[
        "Januar", "Februar", "März", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember"
    ])
    year = st.number_input("Jahr", min_value=2000, max_value=2100, value=datetime.now().year, step=1)
    volume = st.number_input("Volumen", min_value=0.0, value=100.0, step=0.1)
    use_manual = st.checkbox("Manuelles Volumen für Forecast verwenden", value=False)
    submitted = st.form_submit_button("Eintrag hinzufügen")
    if submitted:
        if "manual_volumes" not in st.session_state:
            st.session_state["manual_volumes"] = []
        st.session_state["manual_volumes"].append({
            "Monat": month,
            "Jahr": int(year),
            "Volumen": volume,
            "Manuell für Forecast": use_manual
        })
        st.success("Eintrag hinzugefügt!")

# Interaktive Bearbeitung der manuellen Monatsvolumen
if "manual_volumes" in st.session_state and st.session_state["manual_volumes"]:
    st.write("### Manuelle Monatsvolumen (bearbeitbar)")
    vol_df = pd.DataFrame(st.session_state["manual_volumes"])
    if "Manuell für Forecast" not in vol_df.columns:
        vol_df["Manuell für Forecast"] = False
    vol_df = st.data_editor(vol_df, num_rows="dynamic", key="volumes_editor")
    st.table(vol_df)
    st.session_state["manual_volumes"] = vol_df.to_dict("records")
    
    # Erstelle in vol_df eine Spalte "Month_Year" im Format "YYYY-MM"
    month_map = {
        "Januar": "01", "Februar": "02", "März": "03", "April": "04",
        "Mai": "05", "Juni": "06", "Juli": "07", "August": "08",
        "September": "09", "Oktober": "10", "November": "11", "Dezember": "12"
    }
    vol_df["Month_Year"] = vol_df["Jahr"].astype(str) + "-" + vol_df["Monat"].map(month_map)
    
    # Falls Forecast-Daten vorliegen, aggregiere Forecast-Monatsvolumen
    if "forecast" in st.session_state:
        forecast_df = st.session_state["forecast"]
        last_hist_date = st.session_state["data"]["ds"].max()
        forecast_future = forecast_df[forecast_df["ds"] > last_hist_date]
        if not forecast_future.empty:
            forecast_future["Month_Year"] = forecast_future["ds"].dt.to_period("M").astype(str)
            forecast_monthly = forecast_future.groupby("Month_Year")["yhat"].sum().reset_index()
        else:
            forecast_monthly = pd.DataFrame(columns=["Month_Year", "yhat"])
    else:
        forecast_monthly = pd.DataFrame(columns=["Month_Year", "yhat"])
    
    # Merge manuelle Eingaben mit den Forecast-Werten (Vergleichstabelle)
    merged_vol = pd.merge(vol_df, forecast_monthly, on="Month_Year", how="left")
    merged_vol.rename(columns={"Volumen": "Manuelles Volumen", "yhat": "Forecast Volumen"}, inplace=True)
    merged_vol["Forecast Volumen"] = merged_vol["Forecast Volumen"].fillna("Keine Daten")
    st.write("### Vergleich: Manuelle vs. Forecast Monatsvolumen")
    st.table(merged_vol[["Monat", "Jahr", "Manuelles Volumen", "Forecast Volumen", "Manuell für Forecast"]])

# ─────────────────────────────
# Bereich: Forecast-Horizont (Enddatum)
# ─────────────────────────────
forecast_horizon = None
if "data" in st.session_state and st.session_state["data"] is not None:
    last_date = st.session_state["data"]["ds"].max().date()
    st.sidebar.write("Letztes Datum in den historischen Daten:", last_date)
    forecast_end = st.sidebar.date_input("Enddatum für Forecast-Horizont", value=last_date + timedelta(days=30))
    forecast_horizon = (forecast_end - last_date).days
    if forecast_horizon <= 0:
        st.error("Das Enddatum muss nach dem letzten Datum der historischen Daten liegen.")
else:
    forecast_horizon = DEFAULT_CONFIG["forecast_horizon"]

# ─────────────────────────────
# Bereich: Weitere Forecast-Einstellungen
# ─────────────────────────────
api_key = st.sidebar.text_input("🔑 OpenWeatherMap API Key", type="password")
latitude = st.sidebar.number_input("🌍 Breitengrad", value=52.5200)
longitude = st.sidebar.number_input("🌍 Längengrad", value=13.4050)
changepoint_prior_scale = st.sidebar.slider("🔄 Changepoint Prior Scale", 0.01, 0.5, 0.05)
seasonality_prior_scale = st.sidebar.slider("📊 Seasonality Prior Scale", 0.01, 10.0, 10.0)

# ─────────────────────────────
# Bereich: Forecast-Berechnung und Visualisierung
# ─────────────────────────────
if "data" in st.session_state and st.session_state["data"] is not None and forecast_horizon > 0:
    if st.button("🚀 Forecast starten"):
        with st.spinner("📡 Modell wird trainiert..."):
            model = train_model(st.session_state["data"], changepoint_prior_scale, seasonality_prior_scale)
            future = model.make_future_dataframe(periods=forecast_horizon)
            for col in ['temperature', 'humidity', 'traffic_intensity', 'event_count']:
                if col in st.session_state["data"].columns:
                    if len(st.session_state["data"][col]) >= forecast_horizon:
                        future[col] = st.session_state["data"][col].iloc[-forecast_horizon:].values
                    else:
                        future[col] = np.nan
            forecast = model.predict(future)
            
            # Falls manuelle Monatsvolumen für Forecast aktiviert sind, passe die täglichen Forecast-Werte an
            if "manual_volumes" in st.session_state:
                forecast["Month_Year"] = forecast["ds"].dt.to_period("M").astype(str)
                month_map = {
                    "Januar": "01", "Februar": "02", "März": "03", "April": "04",
                    "Mai": "05", "Juni": "06", "Juli": "07", "August": "08",
                    "September": "09", "Oktober": "10", "November": "11", "Dezember": "12"
                }
                for entry in st.session_state["manual_volumes"]:
                    if entry.get("Manuell für Forecast", False):
                        month_year = f"{entry['Jahr']}-{month_map.get(entry['Monat'], '00')}"
                        mask = forecast["Month_Year"] == month_year
                        if mask.sum() > 0:
                            sum_yhat = forecast.loc[mask, "yhat"].sum()
                            if sum_yhat != 0:
                                factor = entry["Volumen"] / sum_yhat
                                forecast.loc[mask, "yhat"] *= factor
                                if "yhat_lower" in forecast.columns:
                                    forecast.loc[mask, "yhat_lower"] *= factor
                                if "yhat_upper" in forecast.columns:
                                    forecast.loc[mask, "yhat_upper"] *= factor
            st.session_state["forecast"] = forecast
            st.success("✅ Forecast erfolgreich erstellt!")
            
            st.subheader("📉 Forecast-Visualisierung")
            fig = px.line(forecast, x="ds", y="yhat", title="🔮 Prognose")
            fig.add_scatter(x=st.session_state["data"]["ds"], y=st.session_state["data"]["y"], mode="lines", name="Tatsächliche Werte")
            st.plotly_chart(fig)
            
            st.subheader("📊 Prophet-Komponenten")
            components_fig = model.plot_components(forecast)
            st.pyplot(components_fig)
            
            historical = st.session_state["data"][["ds", "y"]]
            merged = pd.merge(historical, forecast[["ds", "yhat"]], on="ds", how="inner")
            if merged.empty:
                st.write("⚠️ Es gibt keine überlappenden Zeitpunkte zwischen den historischen Daten und dem Forecast. Performance-Metriken können nicht berechnet werden.")
            else:
                mae = mean_absolute_error(merged["y"], merged["yhat"])
                mse = mean_squared_error(merged["y"], merged["yhat"])
                r2 = r2_score(merged["y"], merged["yhat"])
                st.write(f"📊 **Metriken:**\n- MAE: {mae:.2f}\n- MSE: {mse:.2f}\n- R²: {r2:.2f}")
            
            csv = forecast.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 CSV herunterladen",
                data=csv,
                file_name="forecast_results.csv",
                mime="text/csv"
            )
