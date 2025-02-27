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

# Setze die Seitenkonfiguration gleich zu Beginn
st.set_page_config(page_title="Forecasting Tool", layout="wide")

# Standard-Konfigurationsdatei
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "forecast_horizon": 30,  # Fallback-Wert, falls keine Daten vorliegen
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

# =============================================================================
# Modul: Daten laden und filtern
# =============================================================================
def load_data():
    """
    Lädt die Daten entweder aus einer Datei (CSV/Excel) oder aus Google Sheets,
    wendet einen optionalen Filter an und gibt den DataFrame zurück.
    """
    try:
        data_source = st.sidebar.radio("Datenquelle", options=["Datei (CSV/Excel)", "Google Sheets"], key="data_source")
        data = None
        if data_source == "Datei (CSV/Excel)":
            uploaded_file = st.sidebar.file_uploader("📂 Datei hochladen (CSV oder Excel)", type=["csv", "xls", "xlsx"])
            if uploaded_file is None:
                return None
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(uploaded_file)
            else:
                st.sidebar.error("Nicht unterstützter Dateityp.")
        elif data_source == "Google Sheets":
            sheet_url = st.sidebar.text_input("Google Sheets URL", value="")
            cred_path = st.sidebar.text_input("Pfad zur Credentials JSON", value="credentials.json")
            if not sheet_url:
                st.info("Bitte geben Sie die Google Sheets URL ein.")
                return None
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(cred_path, scope)
            client = gspread.authorize(creds)
            sheet = client.open_by_url(sheet_url)
            worksheet = sheet.get_worksheet(0)
            data = pd.DataFrame(worksheet.get_all_records())
        if data is None or data.empty:
            st.sidebar.error("❌ Die Datenquelle liefert keine Daten.")
            return None

        columns = list(data.columns)
        ds_col = st.sidebar.selectbox("Wähle das Datums-Feld (ds)", columns, key="ds_col")
        y_col = st.sidebar.selectbox("Wähle das Zielvariable-Feld (y)", columns, key="y_col")
        data['ds'] = pd.to_datetime(data[ds_col], errors='coerce')
        if data['ds'].isnull().all():
            st.sidebar.error("❌ Das ausgewählte Datums-Feld enthält keine gültigen Datumswerte.")
            return None
        data['y'] = data[y_col]
        # Aggregiere untertägige Daten auf Tagesbasis
        data['ds'] = data['ds'].dt.floor('D')
        # Optional: Filterfeld
        filter_field = st.sidebar.selectbox("Filterfeld (optional)", ["Keine Filterung"] + columns, key="filter_field")
        if filter_field != "Keine Filterung":
            unique_vals = sorted(data[filter_field].dropna().unique().tolist())
            filter_value = st.sidebar.selectbox(f"Filterwert für {filter_field}", options=["Alle"] + [str(val) for val in unique_vals], key="filter_value")
            if filter_value != "Alle":
                data = data[data[filter_field].astype(str) == filter_value]
        st.sidebar.success("✅ Daten erfolgreich geladen (ggf. gefiltert)!")
        st.write("### Datenvorschau")
        st.write(data.head())
        return data
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        return None

# =============================================================================
# Modul: Manuelle Monatsvolumen
# =============================================================================
def process_manual_volumes():
    """
    Ermöglicht das Hinzufügen und Bearbeiten manueller Monatsvolumen.
    Gibt ein DataFrame mit den manuellen Einträgen (inkl. Spalte 'Month_Year') zurück.
    """
    try:
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
        if "manual_volumes" in st.session_state and st.session_state["manual_volumes"]:
            st.write("### Manuelle Monatsvolumen (bearbeitbar)")
            vol_df = pd.DataFrame(st.session_state["manual_volumes"])
            if "Manuell für Forecast" not in vol_df.columns:
                vol_df["Manuell für Forecast"] = False
            vol_df = st.data_editor(vol_df, num_rows="dynamic", key="volumes_editor")
            st.table(vol_df)
            st.session_state["manual_volumes"] = vol_df.to_dict("records")
            # Erstelle Spalte "Month_Year"
            month_map = {
                "Januar": "01", "Februar": "02", "März": "03", "April": "04",
                "Mai": "05", "Juni": "06", "Juli": "07", "August": "08",
                "September": "09", "Oktober": "10", "November": "11", "Dezember": "12"
            }
            vol_df["Month_Year"] = vol_df["Jahr"].astype(str) + "-" + vol_df["Monat"].map(month_map)
            return vol_df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Fehler bei der Verarbeitung manueller Volumen: {e}")
        return pd.DataFrame()

# =============================================================================
# Modul: Forecast-Berechnung
# =============================================================================
def build_forecast(data, forecast_horizon, freq_option, changepoint_prior_scale, seasonality_prior_scale):
    try:
        model = train_model(data, changepoint_prior_scale, seasonality_prior_scale)
        future = model.make_future_dataframe(periods=forecast_horizon, freq=freq_option)
        for col in ['temperature', 'humidity', 'traffic_intensity', 'event_count']:
            if col in data.columns:
                if len(data[col]) >= forecast_horizon:
                    future[col] = data[col].iloc[-forecast_horizon:].values
                else:
                    future[col] = np.nan
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        st.error(f"Fehler bei der Forecast-Berechnung: {e}")
        return None, None

# =============================================================================
# Hauptprogramm
# =============================================================================
def main():
    st.title("📈 Intelligentes Forecasting Tool")
    st.sidebar.header("Einstellungen")
    
    # Daten laden
    data = load_data()
    if data is None:
        return
    st.session_state['data'] = data
    
    # Manuelle Monatsvolumen verarbeiten
    vol_df = process_manual_volumes()
    
    # Forecast-Horizont und Frequenz definieren
    try:
        last_date = st.session_state["data"]["ds"].max().date()
        st.sidebar.write("Letztes Datum in den historischen Daten:", last_date)
        forecast_end = st.sidebar.date_input("Enddatum für Forecast-Horizont", value=last_date + timedelta(days=30))
        forecast_horizon = (forecast_end - last_date).days
        if forecast_horizon <= 0:
            st.error("Das Enddatum muss nach dem letzten Datum der historischen Daten liegen.")
            return
    except Exception as e:
        st.error(f"Fehler bei der Bestimmung des Forecast-Horizonts: {e}")
        return
    
    freq_option = st.sidebar.selectbox("Forecast-Frequenz", options=["D", "60min", "30min", "15min"], index=0)
    
    # Weitere Forecast-Einstellungen
    api_key = st.sidebar.text_input("🔑 OpenWeatherMap API Key", type="password")
    latitude = st.sidebar.number_input("🌍 Breitengrad", value=52.5200)
    longitude = st.sidebar.number_input("🌍 Längengrad", value=13.4050)
    changepoint_prior_scale = st.sidebar.slider("🔄 Changepoint Prior Scale", 0.01, 0.5, 0.05)
    seasonality_prior_scale = st.sidebar.slider("📊 Seasonality Prior Scale", 0.01, 10.0, 10.0)
    
    # Forecast starten
    if st.button("🚀 Forecast starten"):
        model, forecast = build_forecast(st.session_state["data"], forecast_horizon, freq_option,
                                         changepoint_prior_scale, seasonality_prior_scale)
        if forecast is None:
            return
        # Speichere den Original-Forecast (ohne manuelle Anpassung) für Vergleichstabelle
        st.session_state["forecast_original"] = forecast.copy()
        
        # Anwenden manueller Anpassungen (dies beeinflusst den Forecast, nicht die Vergleichstabelle)
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
        
        # Visualisierung
        st.subheader("📉 Forecast-Visualisierung")
        fig = px.line(forecast, x="ds", y="yhat", title="🔮 Prognose")
        fig.add_scatter(x=st.session_state["data"]["ds"], y=st.session_state["data"]["y"],
                        mode="lines", name="Tatsächliche Werte")
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
        
        # Vergleichstabelle: Verwende den Original-Forecast (ohne manuelle Skalierung)
        if "forecast_original" in st.session_state:
            forecast_orig = st.session_state["forecast_original"]
            if not forecast_orig.empty and "manual_volumes" in st.session_state:
                forecast_orig["Month_Year"] = forecast_orig["ds"].dt.to_period("M").astype(str)
                vol_df = process_manual_volumes()  # aktuelles DF der manuellen Einträge
                merged_vol = pd.merge(vol_df,
                                        forecast_orig.groupby("Month_Year")["yhat"].sum().reset_index(),
                                        on="Month_Year", how="left")
                merged_vol.rename(columns={"Volumen": "Manuelles Volumen", "yhat": "Forecast Volumen"}, inplace=True)
                merged_vol["Forecast Volumen"] = merged_vol["Forecast Volumen"].fillna("Keine Daten")
                st.write("### Vergleich: Manuelle vs. Forecast Monatsvolumen")
                st.table(merged_vol[["Monat", "Jahr", "Manuelles Volumen", "Forecast Volumen", "Manuell für Forecast"]])

if __name__ == "__main__":
    main()
