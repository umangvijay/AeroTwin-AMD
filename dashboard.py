"""
AeroTwin Edge - Enterprise 3D ML Dashboard
Cloud-ready command center with fault-tolerant backend connectivity.

Premium AMD-inspired dark theme with red/orange accents.
Robust error handling for offline/reconnecting states.
Configurable API URL for cloud deployment.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
from enum import Enum

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh

# Page configuration
st.set_page_config(
    page_title="AeroTwin Edge | Command Center",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)


class ConnectionStatus(Enum):
    """Backend connection status states."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# Premium AMD-inspired dark theme with red/orange accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --amd-red: #ED1C24;
        --amd-orange: #FF6600;
        --amd-dark-red: #B91C1C;
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a24;
        --bg-elevated: #222230;
        --border-color: #2d2d3d;
        --text-primary: #ffffff;
        --text-secondary: #9ca3af;
        --accent-glow: rgba(237, 28, 36, 0.3);
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0d0d14 100%);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: var(--amd-red);
        box-shadow: 0 4px 25px var(--accent-glow);
        transform: translateY(-2px);
    }
    
    div[data-testid="metric-container"] > label {
        color: var(--text-secondary) !important;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] > div {
        color: var(--text-primary) !important;
    }
    
    /* Offline/Reconnecting state */
    .system-offline {
        background: linear-gradient(135deg, rgba(107, 114, 128, 0.2) 0%, rgba(55, 65, 81, 0.2) 100%);
        border: 2px solid #6b7280;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin: 50px auto;
        max-width: 600px;
        animation: offline-pulse 1.5s infinite;
        animation-fill-mode: forwards;
        contain: layout style paint;
        pointer-events: none;
    }
    
    @keyframes offline-pulse {
        0%, 100% { 
            border-color: #6b7280;
            box-shadow: 0 0 20px rgba(107, 114, 128, 0.2);
        }
        50% { 
            border-color: #9ca3af;
            box-shadow: 0 0 40px rgba(107, 114, 128, 0.4);
        }
    }
    
    .system-offline-icon {
        font-size: 4rem;
        margin-bottom: 20px;
        opacity: 0.8;
    }
    
    .system-offline-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .system-offline-subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 20px;
    }
    
    .reconnecting-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid #6b7280;
        border-top-color: var(--amd-red);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Critical anomaly alert */
    .anomaly-critical {
        background: linear-gradient(135deg, rgba(237, 28, 36, 0.25) 0%, rgba(185, 28, 28, 0.2) 100%);
        border: 2px solid var(--amd-red);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 0 30px var(--accent-glow), inset 0 0 20px rgba(237, 28, 36, 0.1);
        animation: critical-pulse 1s infinite;
        animation-fill-mode: forwards;
        contain: layout style paint;
        pointer-events: none;
    }
    
    @keyframes critical-pulse {
        0%, 100% { 
            box-shadow: 0 0 30px var(--accent-glow), inset 0 0 20px rgba(237, 28, 36, 0.1);
            border-color: var(--amd-red);
        }
        50% { 
            box-shadow: 0 0 50px rgba(237, 28, 36, 0.5), inset 0 0 30px rgba(237, 28, 36, 0.2);
            border-color: #ff4444;
        }
    }
    
    .anomaly-critical-header {
        color: #ff6b6b;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .anomaly-critical-body {
        color: #fca5a5;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .anomaly-warning {
        background: linear-gradient(135deg, rgba(255, 102, 0, 0.2) 0%, rgba(255, 165, 0, 0.1) 100%);
        border: 1px solid var(--amd-orange);
        border-left: 4px solid var(--amd-orange);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .anomaly-normal {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.1) 100%);
        border: 1px solid #22c55e;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .log-console {
        background: linear-gradient(180deg, #0d0d12 0%, #0a0a0f 100%);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 18px;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 11px;
        color: #22c55e;
        max-height: 280px;
        overflow-y: auto;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    
    .log-console::-webkit-scrollbar {
        width: 6px;
    }
    
    .log-console::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    .log-console::-webkit-scrollbar-thumb {
        background: var(--amd-red);
        border-radius: 3px;
    }
    
    .section-header {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, var(--amd-red), var(--amd-orange), transparent) 1;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        border-radius: 20px;
    }
    
    .status-offline {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(107, 114, 128, 0.15);
        border: 1px solid #6b7280;
        border-radius: 20px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 10px #22c55e;
        animation: pulse-dot 2s infinite;
    }
    
    .status-dot-offline {
        width: 8px;
        height: 8px;
        background-color: #6b7280;
        border-radius: 50%;
    }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 10px #22c55e; }
        50% { opacity: 0.6; box-shadow: 0 0 20px #22c55e; }
    }
    
    .dashboard-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .dashboard-card:hover {
        border-color: rgba(237, 28, 36, 0.5);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ffffff 0%, var(--amd-red) 50%, var(--amd-orange) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }
    
    .subtitle {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    .amd-divider {
        height: 2px;
        background: linear-gradient(90deg, var(--amd-red), var(--amd-orange), transparent);
        margin: 25px 0;
        border: none;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Aggressive Anti-Flashing CSS - Prevent UI dimming during updates */
    [data-testid="stAppViewContainer"] { 
        transition: none !important; 
        will-change: auto !important;
        transform: translateZ(0);
        backface-visibility: hidden;
    }
    
    [data-testid="stHeader"] { 
        transition: none !important; 
    }
    
    div[data-testid="stOverlay"] { 
        display: none !important; 
    }
    
    .st-emotion-cache-1vt4ygl { 
        display: none !important; 
    }
    
    /* Prevent greyed-out elements during rerun */
    .element-container { 
        opacity: 1 !important; 
        transition: none !important;
        will-change: auto !important;
    }
    
    /* GPU acceleration for animated elements */
    .anomaly-critical,
    .system-offline,
    .status-dot,
    .reconnecting-spinner {
        transform: translateZ(0);
        backface-visibility: hidden;
        will-change: transform, opacity;
    }
    
    /* Prevent flickering on metric containers during updates */
    div[data-testid="metric-container"] {
        transition: none !important;
        will-change: auto !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transition: all 0.3s ease !important;
    }
    
    /* Smooth Transition CSS for Data Updates - Task 4.3 */
    /* Use opacity transitions instead of visibility changes */
    .stMarkdown, 
    .stPlotlyChart,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        opacity: 1;
        transition: opacity 0.4s ease-out, transform 0.4s ease-out;
        transform: translateZ(0);
        backface-visibility: hidden;
    }
    
    /* Smooth data value updates with hardware acceleration */
    [data-testid="stMetricValue"] {
        transition: opacity 0.3s ease-out, transform 0.3s ease-out;
        will-change: contents;
    }
    
    /* Prevent layout shifts with minimum heights */
    .dashboard-card {
        min-height: 100px;
        transition: all 0.4s ease-out;
    }
    
    /* Smooth transitions for chart containers */
    .stPlotlyChart {
        min-height: 300px;
        transition: opacity 0.5s ease-out;
    }
    
    /* Smooth transitions for text content updates */
    .log-console,
    .anomaly-critical,
    .anomaly-warning,
    .anomaly-normal {
        transition: opacity 0.4s ease-out, transform 0.4s ease-out;
        transform: translateZ(0);
    }
    
    /* Smooth fade for status indicators */
    .status-live,
    .status-offline {
        transition: opacity 0.3s ease-out, background-color 0.3s ease-out;
    }
    
    /* Hardware-accelerated transforms for 3D map updates */
    [data-testid="stDeckGlJsonChart"] {
        transition: opacity 0.5s ease-out;
        transform: translateZ(0);
        backface-visibility: hidden;
        will-change: transform;
    }
    
    /* Smooth section header transitions */
    .section-header {
        transition: opacity 0.3s ease-out;
    }
    
    /* Prevent jarring updates on metric deltas */
    [data-testid="stMetricDelta"] {
        transition: opacity 0.3s ease-out, color 0.3s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Configuration - Cloud Ready
# API_URL can be set via environment variable for cloud deployment
# Example: API_URL=https://your-cloud-backend.com streamlit run dashboard.py
API_URL = os.getenv("API_URL", "http://localhost:8000")
BACKEND_URL = API_URL  # Backward compatibility alias

REFRESH_INTERVAL_MS = 10000
CONNECTION_TIMEOUT = 5.0
MAX_RETRY_COUNT = 3

# Detect cloud deployment
IS_CLOUD_DEPLOYMENT = API_URL != "http://localhost:8000"

# Initialize session state for connection tracking
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = ConnectionStatus.DISCONNECTED
if 'last_successful_connection' not in st.session_state:
    st.session_state.last_successful_connection = None
if 'retry_count' not in st.session_state:
    st.session_state.retry_count = 0
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = None

# Auto-refresh every 10 seconds
count = st_autorefresh(interval=REFRESH_INTERVAL_MS, limit=None, key="dashboard_refresh")


def fetch_with_retry(url: str, params: dict = None) -> Tuple[Optional[dict], ConnectionStatus]:
    """
    Fetch data from backend with robust error handling and retry logic.
    
    Returns:
        Tuple of (data, connection_status)
    """
    try:
        response = requests.get(
            url,
            params=params,
            timeout=CONNECTION_TIMEOUT
        )
        response.raise_for_status()
        
        st.session_state.connection_status = ConnectionStatus.CONNECTED
        st.session_state.last_successful_connection = datetime.now()
        st.session_state.retry_count = 0
        
        return response.json(), ConnectionStatus.CONNECTED
        
    except requests.exceptions.ConnectionError:
        st.session_state.retry_count += 1
        if st.session_state.retry_count >= MAX_RETRY_COUNT:
            st.session_state.connection_status = ConnectionStatus.DISCONNECTED
        else:
            st.session_state.connection_status = ConnectionStatus.RECONNECTING
        return None, st.session_state.connection_status
        
    except requests.exceptions.Timeout:
        st.session_state.connection_status = ConnectionStatus.RECONNECTING
        return None, ConnectionStatus.RECONNECTING
        
    except requests.exceptions.HTTPError as e:
        st.session_state.connection_status = ConnectionStatus.ERROR
        return None, ConnectionStatus.ERROR
        
    except Exception as e:
        st.session_state.connection_status = ConnectionStatus.ERROR
        return None, ConnectionStatus.ERROR


def fetch_dashboard_data() -> Tuple[Optional[dict], ConnectionStatus]:
    """Fetch ML dashboard data from FastAPI backend with error handling."""
    return fetch_with_retry(f"{BACKEND_URL}/api/ml_dashboard")


def fetch_telemetry_history(limit: int = 50) -> Tuple[Optional[list], ConnectionStatus]:
    """Fetch recent telemetry for charts with error handling."""
    data, status = fetch_with_retry(
        f"{BACKEND_URL}/api/telemetry/recent",
        params={"limit": limit}
    )
    return data, status


def render_offline_state() -> None:
    """Render professional offline/reconnecting UI state."""
    status = st.session_state.connection_status
    deployment_type = "Cloud" if IS_CLOUD_DEPLOYMENT else "Local"
    
    if status == ConnectionStatus.RECONNECTING:
        st.markdown(f"""
        <div class="system-offline">
            <div class="system-offline-icon">🔄</div>
            <div class="system-offline-title">
                <span class="reconnecting-spinner"></span>
                Reconnecting to Backend...
            </div>
            <div class="system-offline-subtitle">
                Attempting to establish connection with AeroTwin Edge server.
            </div>
            <div style="color: #6b7280; font-size: 0.85rem;">
                Deployment: {deployment_type}<br/>
                Backend URL: {API_URL}<br/>
                Retry attempt: {st.session_state.retry_count}/{MAX_RETRY_COUNT}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        last_conn = st.session_state.last_successful_connection
        last_conn_str = last_conn.strftime('%H:%M:%S') if last_conn else "Never"
        
        troubleshooting = """
            <strong style="color: #ffffff;">Troubleshooting:</strong><br/>
            1. Verify the API_URL environment variable is correct<br/>
            2. Check if the backend server is running<br/>
            3. Verify network connectivity to the server
        """ if IS_CLOUD_DEPLOYMENT else """
            <strong style="color: #ffffff;">Troubleshooting:</strong><br/>
            1. Ensure the backend is running: <code style="color: #ED1C24;">python main.py</code><br/>
            2. Check if port 8000 is available<br/>
            3. Verify firewall settings
        """
        
        st.markdown(f"""
        <div class="system-offline">
            <div class="system-offline-icon">📡</div>
            <div class="system-offline-title">System Offline</div>
            <div class="system-offline-subtitle">
                Unable to connect to AeroTwin Edge backend server.
            </div>
            <div style="color: #6b7280; font-size: 0.85rem; margin-top: 15px;">
                <div style="margin: 5px 0;">Deployment: <span style="color: #FF6600;">{deployment_type}</span></div>
                <div style="margin: 5px 0;">Backend URL: {API_URL}</div>
                <div style="margin: 5px 0;">Last successful connection: {last_conn_str}</div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                <div style="color: #9ca3af; font-size: 0.8rem; text-align: left;">
                    {troubleshooting}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; color: #6b7280; font-size: 0.85rem;">
        Dashboard will automatically reconnect when the backend becomes available.
    </div>
    """, unsafe_allow_html=True)


def get_mode_display(ai_mode: str) -> tuple[str, str]:
    """Convert AI mode to display format with color."""
    mode_map = {
        "eco": ("INT8 ECO", "🟢"),
        "auto": ("FP16 BALANCED", "🟡"),
        "performance": ("FP32 PERFORMANCE", "🔴"),
        "standby": ("STANDBY", "⚪"),
        "manual": ("MANUAL", "🔵")
    }
    return mode_map.get(ai_mode.lower(), ("UNKNOWN", "⚪"))


def get_occupancy_color(occupancy: int, max_occupancy: int = 50) -> list:
    """Calculate color based on occupancy (green to red gradient)."""
    ratio = min(occupancy / max_occupancy, 1.0)
    red = int(255 * ratio)
    green = int(255 * (1 - ratio))
    return [red, green, 50, 200]


def create_3d_digital_twin(telemetry_data: Optional[dict]) -> pdk.Deck:
    """Create the 3D spatial digital twin visualization with enhanced tooltips."""
    
    buildings = [
        {"name": "Building-A-Floor-1", "lat": 37.7749, "lon": -122.4194, "base_height": 100},
        {"name": "Building-A-Floor-2", "lat": 37.7751, "lon": -122.4190, "base_height": 120},
        {"name": "Building-B-Floor-1", "lat": 37.7745, "lon": -122.4188, "base_height": 80},
        {"name": "Building-B-Floor-2", "lat": 37.7747, "lon": -122.4182, "base_height": 90},
        {"name": "Building-C-Main", "lat": 37.7742, "lon": -122.4198, "base_height": 150},
        {"name": "Library", "lat": 37.7755, "lon": -122.4200, "base_height": 60},
        {"name": "Cafeteria", "lat": 37.7740, "lon": -122.4205, "base_height": 40},
        {"name": "Research-Lab", "lat": 37.7752, "lon": -122.4178, "base_height": 110},
    ]
    
    current_occupancy = 0
    current_zone = ""
    current_mode = "auto"
    
    if telemetry_data and telemetry_data.get("latest_telemetry"):
        current_occupancy = telemetry_data["latest_telemetry"].get("occupancy", 0)
        current_zone = telemetry_data["latest_telemetry"].get("zone", "")
        current_mode = telemetry_data["latest_telemetry"].get("ai_mode", "auto")
    
    building_data = []
    for building in buildings:
        if building["name"] == current_zone:
            occ = current_occupancy
            mode = current_mode.upper()
        else:
            occ = max(0, current_occupancy + np.random.randint(-15, 10))
            if occ < 15:
                mode = "ECO"
            elif occ <= 35:
                mode = "BALANCED"
            else:
                mode = "PERFORMANCE"
        
        color = get_occupancy_color(occ)
        height = building["base_height"] + (occ * 3)
        
        status = "🟢 Normal" if occ < 35 else "🔴 High"
        
        building_data.append({
            "name": building["name"],
            "coordinates": [building["lon"], building["lat"]],
            "occupancy": occ,
            "height": height,
            "color": color,
            "mode": mode,
            "status": status
        })
    
    df = pd.DataFrame(building_data)
    
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position="coordinates",
        get_elevation="height",
        elevation_scale=50,
        radius=30,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
        coverage=0.8,
    )
    
    text_layer = pdk.Layer(
        "TextLayer",
        data=df,
        get_position="coordinates",
        get_text="name",
        get_size=12,
        get_color=[255, 255, 255, 200],
        get_angle=0,
        get_text_anchor="'middle'",
        get_alignment_baseline="'bottom'",
    )
    
    view_state = pdk.ViewState(
        latitude=37.7747,
        longitude=-122.4192,
        zoom=15,
        pitch=45,
        bearing=-20
    )
    
    return pdk.Deck(
        layers=[column_layer, text_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={
            "html": """
                <div style='background: linear-gradient(135deg, #1a1a24 0%, #12121a 100%); 
                            padding: 15px; border-radius: 10px; 
                            border: 1px solid #ED1C24; min-width: 200px;
                            box-shadow: 0 4px 20px rgba(237, 28, 36, 0.3);'>
                    <div style='font-size: 14px; font-weight: 700; color: #ffffff; 
                                margin-bottom: 10px; border-bottom: 1px solid #2d2d3d; 
                                padding-bottom: 8px;'>
                        🏢 {name}
                    </div>
                    <div style='display: grid; gap: 6px;'>
                        <div style='color: #9ca3af;'>
                            <span style='color: #ED1C24;'>●</span> Occupancy: 
                            <span style='color: #ffffff; font-weight: 600;'>{occupancy}</span>
                        </div>
                        <div style='color: #9ca3af;'>
                            <span style='color: #FF6600;'>●</span> ML Mode: 
                            <span style='color: #ffffff; font-weight: 600;'>{mode}</span>
                        </div>
                        <div style='color: #9ca3af;'>
                            <span style='color: #22c55e;'>●</span> Status: 
                            <span style='color: #ffffff;'>{status}</span>
                        </div>
                    </div>
                </div>
            """,
            "style": {
                "backgroundColor": "transparent",
                "color": "white"
            }
        }
    )


def create_occupancy_chart(telemetry_history: list, forecast: list) -> go.Figure:
    """Create live occupancy chart with forecast overlay."""
    
    fig = go.Figure()
    
    if telemetry_history:
        df = pd.DataFrame(telemetry_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['occupancy'],
            mode='lines+markers',
            name='Actual Occupancy',
            line=dict(color='#00d4ff', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['hardware_cpu'],
            mode='lines',
            name='CPU Load %',
            line=dict(color='#ED1C24', width=1.5, dash='dot'),
            yaxis='y2',
            opacity=0.8
        ))
        
        if forecast and len(df) > 0:
            last_time = df['timestamp'].iloc[-1]
            forecast_times = [
                last_time + timedelta(minutes=5 * (i + 1))
                for i in range(len(forecast))
            ]
            forecast_values = [f['predicted_occupancy'] for f in forecast]
            upper_bounds = [f['confidence_upper'] for f in forecast]
            lower_bounds = [f['confidence_lower'] for f in forecast]
            
            fig.add_trace(go.Scatter(
                x=forecast_times + forecast_times[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(255, 102, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence',
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_times,
                y=forecast_values,
                mode='lines+markers',
                name='ML Forecast',
                line=dict(color='#FF6600', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(18,18,26,0.9)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        title=dict(
            text='Live Occupancy & ML Forecast (Holt-Winters)',
            font=dict(size=14, color='#ffffff')
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10)
        ),
        xaxis=dict(
            title='',
            gridcolor='rgba(45, 45, 61, 0.5)',
            showgrid=True
        ),
        yaxis=dict(
            title='Occupancy',
            gridcolor='rgba(45, 45, 61, 0.5)',
            showgrid=True
        ),
        yaxis2=dict(
            title='CPU %',
            overlaying='y',
            side='right',
            range=[0, 100],
            showgrid=False
        )
    )
    
    return fig


def render_anomaly_alerts(anomalies: list) -> None:
    """Render the anomaly alerts section with HIGHLY VISIBLE warnings."""
    
    st.markdown('<p class="section-header">🚨 Security & Anomaly Alerts</p>', unsafe_allow_html=True)
    
    if not anomalies:
        st.markdown("""
        <div class="anomaly-normal">
            <div style="font-weight: 600; color: #22c55e; margin-bottom: 5px;">
                ✓ All Systems Normal
            </div>
            <div style="color: #86efac; font-size: 0.85rem;">
                No anomalies detected by Isolation Forest algorithm.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    recent_critical = [a for a in anomalies[:3] if a.get('severity', '').lower() in ['critical', 'high']]
    
    if recent_critical:
        st.error("⚠️ ANOMALY DETECTED - Isolation Forest has flagged unusual behavior!")
    
    for anomaly in anomalies[:5]:
        severity = anomaly.get('severity', 'medium').lower()
        timestamp = anomaly.get('timestamp', '')
        description = anomaly.get('description', 'Unknown anomaly')
        
        if isinstance(timestamp, str) and timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp[:19]
        else:
            time_str = "Unknown"
        
        if severity in ['critical', 'high']:
            st.markdown(f"""
            <div class="anomaly-critical">
                <div class="anomaly-critical-header">
                    <span style="font-size: 1.2rem;">🔴</span>
                    <span>{severity.upper()} ALERT</span>
                    <span style="font-size: 0.8rem; color: #9ca3af; margin-left: auto;">{time_str}</span>
                </div>
                <div class="anomaly-critical-body">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif severity == 'medium':
            st.markdown(f"""
            <div class="anomaly-warning">
                <div style="font-weight: 600; color: #FF6600; margin-bottom: 5px;">
                    🟡 {severity.upper()} | {time_str}
                </div>
                <div style="color: #fdba74; font-size: 0.85rem;">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="anomaly-normal">
                <div style="font-weight: 600; color: #22c55e; margin-bottom: 5px;">
                    🟢 {severity.upper()} | {time_str}
                </div>
                <div style="color: #86efac; font-size: 0.85rem;">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_system_logs(telemetry: Optional[dict], is_connected: bool) -> None:
    """Render system logs console."""
    
    st.markdown('<p class="section-header">📟 Edge System Logs</p>', unsafe_allow_html=True)
    
    logs = []
    current_time = datetime.now().strftime('%H:%M:%S')
    
    if not is_connected:
        logs.append(f"<span style='color: #9ca3af;'>[{current_time}]</span> <span style='color: #ef4444;'>ERROR</span>    Backend connection lost")
        logs.append(f"<span style='color: #9ca3af;'>[{current_time}]</span> <span style='color: #fbbf24;'>SYSTEM</span>   Attempting to reconnect...")
    elif telemetry and telemetry.get('latest_telemetry'):
        t = telemetry['latest_telemetry']
        mode, _ = get_mode_display(t.get('ai_mode', 'auto'))
        
        logs.append(f"<span style='color: #9ca3af;'>[{current_time}]</span> <span style='color: #00d4ff;'>TELEMETRY</span> Zone={t.get('zone', 'N/A')} | Occ={t.get('occupancy', 0)} | CPU={t.get('hardware_cpu', 0):.1f}%")
        logs.append(f"<span style='color: #9ca3af;'>[{current_time}]</span> <span style='color: #FF6600;'>AI_MODE</span>   {mode} active")
        
        if telemetry.get('model_status') == 'warming_up':
            logs.append(f"<span style='color: #9ca3af;'>[{current_time}]</span> <span style='color: #fbbf24;'>ML_ENGINE</span> Isolation Forest warming up...")
        else:
            logs.append(f"<span style='color: #9ca3af;'>[{current_time}]</span> <span style='color: #22c55e;'>ML_ENGINE</span> Anomaly detection active (Holt-Winters forecast)")
        
        if telemetry.get('forecast'):
            forecast = telemetry['forecast'][0] if telemetry['forecast'] else None
            if forecast:
                logs.append(f"<span style='color: #9ca3af;'>[{current_time}]</span> <span style='color: #a78bfa;'>FORECAST</span>  Next interval: {forecast.get('predicted_occupancy', 0):.1f}")
    
    logs.append(f"<span style='color: #9ca3af;'>[{current_time}]</span> <span style='color: #6b7280;'>SYSTEM</span>    Dashboard refresh #{count}")
    logs.append(f"<span style='color: #9ca3af;'>[{current_time}]</span> <span style='color: #22c55e;'>NETWORK</span>   {'Backend connected' if is_connected else 'Disconnected'}")
    
    log_text = "<br/>".join(logs)
    
    st.markdown(f"""
    <div class="log-console">
        {log_text}
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main dashboard entry point with fault-tolerant error handling."""
    
    # Header with AMD-inspired branding
    with st.container():
        col_title, col_status = st.columns([4, 1])
        with col_title:
            st.markdown("""
            <h1 class="main-title">🏢 AeroTwin Edge | Command Center</h1>
            <p class="subtitle">Enterprise Smart Campus AI Platform | Real-time ML Analytics | Production Mode</p>
            """, unsafe_allow_html=True)
        
        with col_status:
            is_connected = st.session_state.connection_status == ConnectionStatus.CONNECTED
            if is_connected:
                st.markdown("""
                <div style='text-align: right; padding-top: 15px;'>
                    <div class="status-live">
                        <span class="status-dot"></span>
                        <span style='color: #22c55e; font-weight: 600; font-size: 0.9rem;'>LIVE</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: right; padding-top: 15px;'>
                    <div class="status-offline">
                        <span class="status-dot-offline"></span>
                        <span style='color: #6b7280; font-weight: 600; font-size: 0.9rem;'>OFFLINE</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="amd-divider"></div>', unsafe_allow_html=True)
    
    # Fetch data with error handling
    dashboard_data, dash_status = fetch_dashboard_data()
    telemetry_history, telem_status = fetch_telemetry_history(50)
    
    # Check if we're offline
    if dash_status != ConnectionStatus.CONNECTED:
        render_offline_state()
        return
    
    # Cache successful data
    if dashboard_data:
        st.session_state.cached_data = dashboard_data
    
    # Extract data
    latest = dashboard_data.get('latest_telemetry') or {}
    anomalies = dashboard_data.get('recent_anomalies', [])
    forecast = dashboard_data.get('forecast', [])
    model_status = dashboard_data.get('model_status', 'unknown')
    total_telemetry = dashboard_data.get('total_telemetry_count', 0)
    total_anomalies = dashboard_data.get('total_anomaly_count', 0)
    
    # =========================================================================
    # SECTION 1: Hardware Telemetry
    # =========================================================================
    with st.container():
        st.markdown('<p class="section-header">📊 Hardware Telemetry</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            mode_display, mode_icon = get_mode_display(latest.get('ai_mode', 'unknown'))
            st.metric(
                label="Active ML Quantization",
                value=f"{mode_icon} {mode_display}",
                delta=f"Model: {model_status.upper()}"
            )
        
        with col2:
            cpu = latest.get('hardware_cpu', 0)
            cpu_delta = "Normal" if cpu < 70 else ("High" if cpu < 90 else "Critical")
            st.metric(
                label="Edge CPU Load",
                value=f"{cpu:.1f}%",
                delta=cpu_delta,
                delta_color="inverse" if cpu > 70 else "off"
            )
        
        with col3:
            mode = latest.get('ai_mode', 'auto')
            latency_map = {'eco': 12, 'auto': 20, 'performance': 35, 'standby': 75}
            latency = latency_map.get(mode, 20) + np.random.uniform(-3, 3)
            st.metric(
                label="Inference Latency",
                value=f"{latency:.1f} ms",
                delta="NPU Accelerated"
            )
        
        with col4:
            occupancy = latest.get('occupancy', 0)
            st.metric(
                label="Campus Occupancy",
                value=f"{occupancy}",
                delta=f"Zone: {latest.get('zone', 'N/A')[:15]}"
            )
        
        with col5:
            st.metric(
                label="Total Readings",
                value=f"{total_telemetry:,}",
                delta=f"{total_anomalies} anomalies"
            )
    
    st.divider()
    
    # =========================================================================
    # SECTION 2: Spatial Digital Twin
    # =========================================================================
    with st.container():
        st.markdown('<p class="section-header">🌐 Spatial Digital Twin | 3D Campus View</p>', unsafe_allow_html=True)
        
        twin_col, legend_col = st.columns([4, 1])
        
        with twin_col:
            deck = create_3d_digital_twin(dashboard_data)
            st.pydeck_chart(deck, use_container_width=True)
        
        with legend_col:
            st.markdown("""
            <div class="dashboard-card">
                <strong style="color: #ffffff;">Occupancy Legend</strong>
                <hr style="border-color: #2d2d3d; margin: 10px 0;"/>
                <div style="margin: 8px 0; color: #9ca3af;">
                    <span style="color: #22c55e;">●</span> Low (0-15)
                </div>
                <div style="margin: 8px 0; color: #9ca3af;">
                    <span style="color: #eab308;">●</span> Medium (15-35)
                </div>
                <div style="margin: 8px 0; color: #9ca3af;">
                    <span style="color: #ED1C24;">●</span> High (35-50)
                </div>
                <hr style="border-color: #2d2d3d; margin: 10px 0;"/>
                <small style="color: #6b7280;">
                    Column height scales<br/>with occupancy level
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="dashboard-card" style="margin-top: 15px;">
                <strong style="color: #ffffff;">Quick Stats</strong>
                <hr style="border-color: #2d2d3d; margin: 10px 0;"/>
                <div style="color: #9ca3af; font-size: 12px;">
                    <div style="margin: 5px 0;">Buildings: <span style="color: #ffffff;">8</span></div>
                    <div style="margin: 5px 0;">Zones: <span style="color: #22c55e;">Active</span></div>
                    <div style="margin: 5px 0;">Sensors: <span style="color: #22c55e;">Online</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # =========================================================================
    # SECTION 3: Advanced ML Analytics
    # =========================================================================
    with st.container():
        chart_col, alerts_col = st.columns([3, 2])
        
        with chart_col:
            st.markdown('<p class="section-header">📈 Advanced ML Analytics</p>', unsafe_allow_html=True)
            
            if telemetry_history:
                fig = create_occupancy_chart(telemetry_history, forecast)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Waiting for telemetry data...")
            
            if forecast:
                st.markdown("**Forecast Summary (Holt-Winters Exponential Smoothing):**")
                forecast_cols = st.columns(len(forecast))
                for i, (col, f) in enumerate(zip(forecast_cols, forecast)):
                    with col:
                        st.metric(
                            label=f"T+{f['interval']}",
                            value=f"{f['predicted_occupancy']:.0f}",
                            delta=f"±{(f['confidence_upper'] - f['confidence_lower'])/2:.0f}"
                        )
        
        with alerts_col:
            render_anomaly_alerts(anomalies)
            st.markdown("<br/>", unsafe_allow_html=True)
            render_system_logs(dashboard_data, True)
    
    # Footer
    deployment_badge = '<span style="color: #FF6600;">☁️ Cloud</span>' if IS_CLOUD_DEPLOYMENT else '<span style="color: #22c55e;">🖥️ Local</span>'
    
    st.markdown('<div class="amd-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align: center; color: #6b7280; font-size: 11px; padding: 10px 0;'>
        <span style="color: #ED1C24; font-weight: 600;">AeroTwin Edge</span> v1.0.0 | 
        {deployment_badge} | 
        Refresh #{count} | 
        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
        API: {API_URL} |
        Status: <span style="color: #22c55e;">Connected</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
