import streamlit as st
import pandas as pd
import requests
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(
    page_title="Predictive Weather Analytics",
    page_icon="🌤️",
    layout="wide"
)


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Canvas Background Override */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #EEF2F6 0%, #F0FDF4 50%, #FFFBEB 100%) !important;
        background-attachment: fixed;
    }

    /* Core Glassmorphic Card Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 32px 0 rgba(148, 163, 184, 0.05) !important;
        margin-bottom: 1.25rem !important;
        color: #1E293B !important;
    }

    /* Typography Hierarchy */
    .app-title {
        color: #1E293B !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.5rem !important;
    }

    .app-subtitle {
        color: #64748B !important;
        font-weight: 400 !important;
        font-size: 1.05rem !important;
        margin-bottom: 1.5rem !important;
    }

    .section-header {
        color: #1E293B !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
        letter-spacing: -0.01em !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* Label text color overrides */
    label {
        color: #1E293B !important;
        font-weight: 600 !important;
    }

    /* Glassmorphic Form Inputs - Neutralizing System Overrides */
    div[data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.55) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 4px rgba(148, 163, 184, 0.02) !important;
    }

    div[data-baseweb="input"] input {
        color: #1E293B !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }

    /* Custom Pastel Accent Sidebar Run Button */
    div.stButton > button {
        background: linear-gradient(135deg, #E0E7FF 0%, #EEF2F6 100%) !important;
        color: #4F46E5 !important;
        border: 1px solid rgba(199, 210, 254, 0.8) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(148, 163, 184, 0.08) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #EEF2F6 0%, #E0E7FF 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(148, 163, 184, 0.12) !important;
    }

    /* High-Visibility Custom Glassmorphic Table */
    .pastel-table {
        width: 100% !important;
        border-collapse: collapse !important;
        background-color: rgba(255, 255, 255, 0.4) !important;
        color: #1E293B !important;
    }
    
    .pastel-table th {
        background-color: rgba(255, 255, 255, 0.6) !important;
        color: #1E293B !important;
        font-weight: 700 !important;
        text-align: left !important;
        padding: 14px 16px !important;
        font-size: 0.9rem !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.8) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    .pastel-table td {
        padding: 16px 16px !important;
        font-size: 0.95rem !important;
        color: #1E293B !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.3) !important;
        background-color: transparent !important;
    }
    
    .pastel-table tr:last-child td {
        border-bottom: none !important;
    }
    
    .pastel-table tr:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    hr {
        border-top: 1px solid rgba(255, 255, 255, 0.5) !important;
    }

    /* Hiding Deploy Button and Streamlit Default Overlays */
    .stAppDeployButton {
        display: none !important;
    }
    header {
        visibility: hidden !important;
        display: none !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=86400)
def fetch_coordinates(city_name):
    """Fetches latitude, longitude, and formatted name for a given city name."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "results" not in data:
            return None, None, None
        result = data["results"][0]
        resolved_name = f"{result['name']}, {result.get('country', '')}"
        return result["latitude"], result["longitude"], resolved_name
    except Exception as e:
        st.error(f"Geocoding Error: {e}")
        return None, None, None

@st.cache_data(ttl=1800)
def fetch_current_diagnostics(lat, lon):
    """Fetches rich real-time current weather diagnostics, sunrise, sunset, humidity, and rain chance."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["relative_humidity_2m", "surface_pressure"],
        "daily": ["sunrise", "sunset", "precipitation_probability_max"],
        "timezone": "auto",
        "forecast_days": 1
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Diagnostics Error: {e}")
        return None

@st.cache_data(ttl=86400)
def fetch_weather_data(lat, lon):
    """Fetches historical temperature data for the last 2 years."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=730)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max",
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "temperature_2m_max": data["daily"]["temperature_2m_max"]
        })
        return df
    except Exception as e:
        st.error(f"Data Extraction Error: {e}")
        return None


def train_prediction_model(df):
    """Engineers features and trains a Random Forest model."""
    df_model = df.copy()
    df_model['month'] = df_model['date'].dt.month
    df_model['day_of_year'] = df_model['date'].dt.dayofyear
    df_model['yesterday_temp'] = df_model['temperature_2m_max'].shift(1)
    
    df_model = df_model.dropna()
    
    X = df_model[['month', 'day_of_year', 'yesterday_temp']]
    y = df_model['temperature_2m_max']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    last_temp = df['temperature_2m_max'].iloc[-1]
    return model, last_temp


st.markdown("<div class='app-title'>🌤️ Predictive Weather Analytics</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>A premium machine learning platform for historical and predictive forecasting.</div>", unsafe_allow_html=True)


st.markdown("<div class='section-header'>🎛️ Analytics Control Panel</div>", unsafe_allow_html=True)
with st.container():
    col_input, col_slider, col_trigger = st.columns([2, 2, 1])
    with col_input:
        city_input = st.text_input("Enter City Name:", value="Salem")
    with col_slider:
        forecast_horizon = st.slider("Forecast Horizon (Days):", 1, 7, 5)
    with col_trigger:
        # Align button vertically with inputs
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("Run Analytics Pipeline")

st.markdown("<hr>", unsafe_allow_html=True)


if run_btn:
    with st.spinner(f"Analyzing climate patterns for {city_input}..."):
        lat, lon, full_city_name = fetch_coordinates(city_input)
        
        if lat is None:
            st.error("Could not locate the specified city. Please check the spelling.")
        else:
            df = fetch_weather_data(lat, lon)
            diagnostics = fetch_current_diagnostics(lat, lon)
            
            if df is not None and diagnostics is not None:
                # ML Model Training
                model, last_known_temp = train_prediction_model(df)
                
                # Retrieve details of the most recent recorded day
                last_record_date = df['date'].iloc[-1].strftime('%A, %b %d, %Y')
                
                # Parse Live Diagnostics
                live_humidity = diagnostics["current"]["relative_humidity_2m"]
                live_pressure = diagnostics["current"]["surface_pressure"]
                live_sunrise = datetime.datetime.fromisoformat(diagnostics["daily"]["sunrise"][0]).strftime("%H:%M")
                live_sunset = datetime.datetime.fromisoformat(diagnostics["daily"]["sunset"][0]).strftime("%H:%M")
                live_rain_prob = diagnostics["daily"]["precipitation_probability_max"][0]
                
                # Display Header Card
                st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size: 1.6rem; font-weight: 700; color: #1E293B !important;">
                            {full_city_name}
                        </div>
                        <div style="font-size: 1.15rem; color: #4F46E5 !important; font-weight: 600; margin-top: 0.35rem;">
                            Latest Recorded Day ({last_record_date}): {last_known_temp}°C
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Atmospheric Diagnostics Dashboard (2x3 Grid)
                st.markdown("<div class='section-header'>📊 Live Atmospheric Diagnostics</div>", unsafe_allow_html=True)
                
                # Grid Row 1
                row1_col1, row1_col2, row1_col3 = st.columns(3)
                with row1_col1:
                    st.markdown(f"""
                        <div class="glass-card" style="padding: 1.25rem !important;">
                            <div style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Recorded Temperature</div>
                            <div style="font-size: 1.65rem; color: #1E293B; font-weight: 700; margin-top: 0.2rem;">{last_known_temp}°C</div>
                        </div>
                    """, unsafe_allow_html=True)
                with row1_col2:
                    st.markdown(f"""
                        <div class="glass-card" style="padding: 1.25rem !important;">
                            <div style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Chance of Rain</div>
                            <div style="font-size: 1.65rem; color: #1E293B; font-weight: 700; margin-top: 0.2rem;">{live_rain_prob}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                with row1_col3:
                    st.markdown(f"""
                        <div class="glass-card" style="padding: 1.25rem !important;">
                            <div style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Relative Humidity</div>
                            <div style="font-size: 1.65rem; color: #1E293B; font-weight: 700; margin-top: 0.2rem;">{live_humidity}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Grid Row 2
                row2_col1, row2_col2, row2_col3 = st.columns(3)
                with row2_col1:
                    st.markdown(f"""
                        <div class="glass-card" style="padding: 1.25rem !important;">
                            <div style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Surface Pressure</div>
                            <div style="font-size: 1.65rem; color: #1E293B; font-weight: 700; margin-top: 0.2rem;">{live_pressure} hPa</div>
                        </div>
                    """, unsafe_allow_html=True)
                with row2_col2:
                    st.markdown(f"""
                        <div class="glass-card" style="padding: 1.25rem !important;">
                            <div style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Sunrise Time</div>
                            <div style="font-size: 1.65rem; color: #1E293B; font-weight: 700; margin-top: 0.2rem;">{live_sunrise} AM</div>
                        </div>
                    """, unsafe_allow_html=True)
                with row2_col3:
                    st.markdown(f"""
                        <div class="glass-card" style="padding: 1.25rem !important;">
                            <div style="font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Sunset Time</div>
                            <div style="font-size: 1.65rem; color: #1E293B; font-weight: 700; margin-top: 0.2rem;">{live_sunset} PM</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Historical Trend Graph Sliced for 7-Day Showcase
                st.markdown("<div class='section-header'>📈 Historical 7-Day Temperature Trend</div>", unsafe_allow_html=True)
                
                df_last_7 = df.tail(7).copy()
                dates_7 = df_last_7['date'].dt.strftime("%A, %b %d")
                temps_7 = df_last_7['temperature_2m_max']

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates_7, y=temps_7, mode='lines+markers', name='Max Temp (°C)',
                    line=dict(color='#60A5FA', width=4),
                    marker=dict(size=9, color='#2563EB', symbol='circle'),
                    fill='tonexty', fillcolor='rgba(147, 197, 253, 0.06)'
                ))

                fig.update_layout(
                    template='plotly_white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(
                        family="Plus Jakarta Sans, sans-serif",
                        size=12,
                        color="#1E293B"
                    ),
                    margin=dict(l=15, r=15, t=15, b=15),
                    hovermode="x unified",
                    xaxis=dict(
                        showgrid=False,
                        tickfont=dict(color="#1E293B", size=11)
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(30, 41, 59, 0.08)',
                        tickfont=dict(color="#1E293B", size=11),
                        title="Recorded Max Temp (°C)"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                # Predictive Weather Forecast Layout
                st.markdown(f"<div class='section-header'>🔮 Predictive Weather Forecast (Next {forecast_horizon} Days)</div>", unsafe_allow_html=True)
                
                predictions = []
                current_temp = last_known_temp
                current_date = df['date'].iloc[-1]
                
                for i in range(1, forecast_horizon + 1):
                    next_date = current_date + datetime.timedelta(days=i)
                    
                    features = pd.DataFrame({
                        'month': [next_date.month],
                        'day_of_year': [next_date.dayofyear],
                        'yesterday_temp': [current_temp]
                    })
                    
                    pred = model.predict(features)[0]
                    predictions.append({
                        'Day': next_date.strftime('%A'),
                        'Date': next_date.strftime('%b %d, %Y'),
                        'Predicted Temp (°C)': f"{round(pred, 2)}°C"
                    })
                    
                    current_temp = pred
                
                # Format table lines to ensure zero Markdown preformatted code conflicts
                table_rows = ""
                for row in predictions:
                    table_rows += f"<tr><td style='font-weight:700 !important; color:#1E293B !important;'>{row['Day']}</td><td style='color:#475569 !important;'>{row['Date']}</td><td style='font-weight:700 !important; color:#4F46E5 !important;'>{row['Predicted Temp (°C)']}</td></tr>"

                # Constructed layout with strictly trimmed single-line elements to enforce rendering
                html_table = f'<div class="glass-card" style="padding:0px !important; overflow:hidden !important; border-radius:16px !important; border:1px solid rgba(255,255,255,0.6) !important;"><table class="pastel-table"><thead><tr><th>Forecast Day</th><th>Calendar Date</th><th>ML Predicted Temperature</th></tr></thead><tbody>{table_rows}</tbody></table></div>'
                
                # Render using the safest non-markdown parsing pipeline
                st.markdown(html_table, unsafe_allow_html=True)
                
            else:
                st.warning("Failed to retrieve weather data.")
else:
    st.info("Please configure the target city in the Control Panel above and click 'Run Analytics Pipeline' to run predictions.")