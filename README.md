# AeroTwin Edge - Smart Campus AI Platform

Enterprise-grade Smart Campus AI platform with ML-powered anomaly detection, time-series forecasting, and cloud-ready deployment.

## Features

- **Real-time Telemetry Ingestion**: REST API with API key authentication
- **ML Anomaly Detection**: Isolation Forest algorithm running every 10 seconds
- **Time-Series Forecasting**: Holt-Winters Exponential Smoothing for occupancy predictions
- **3D Digital Twin Dashboard**: Interactive pydeck visualization with AMD-themed UI
- **Cloud-Ready**: Bundled simulator for 24/7 cloud deployment without external dependencies

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
cd "AeroTwin AWD"
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python main.py
```

The server starts with:
- REST API at http://localhost:8000
- Bundled edge simulator (generates telemetry automatically)
- ML anomaly detection background task

### 3. Run the Dashboard

```bash
streamlit run dashboard.py
```

Access the command center at http://localhost:8501

## Cloud Deployment (24/7 Judge Evaluation)

### Single Service Deployment

The backend now includes a **bundled simulator** that runs automatically on startup. No need to deploy multiple services!

```bash
# Just deploy main.py - it includes everything
python main.py
```

When deployed to cloud, the system will:
1. Log: `[System] Cloud Deployment Detected. Simulating AMD Ryzen AI edge telemetry.`
2. Generate realistic AMD Ryzen AI metrics (CPU 15-85% based on mode)
3. Trigger demo spikes every 15 cycles for guaranteed anomaly detection

### Environment Variables for Cloud

```bash
# Required
DATABASE_URL=sqlite:///./aerotwin_prod.db
API_KEY=your_secure_key_here

# Optional (defaults shown)
CLOUD_MODE=true
SIMULATOR_ENABLED=true
SIMULATOR_INTERVAL=5.0
DEMO_SPIKE_INTERVAL=15
```

### Dashboard Cloud Configuration

When deploying the dashboard separately (e.g., Streamlit Cloud):

```bash
# Point to your cloud backend
API_URL=https://your-backend-url.com streamlit run dashboard.py
```

Or set in Streamlit Cloud secrets:
```toml
# .streamlit/secrets.toml
API_URL = "https://your-backend-url.com"
```

## API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/api/health` | GET | Comprehensive health check |
| `/api/simulator/status` | GET | Simulator status and cycle count |

### Telemetry (Requires API Key)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/telemetry` | POST | Ingest telemetry (requires `X-API-Key` header) |
| `/api/telemetry/recent` | GET | Get recent telemetry records |

### ML Dashboard

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ml_dashboard` | GET | Dashboard data with forecast |
| `/api/anomalies` | GET | Anomaly records |

## Architecture

```
AeroTwin AWD/
├── main.py           # FastAPI + bundled cloud simulator
├── config.py         # Pydantic-settings configuration
├── database.py       # SQLAlchemy setup
├── models.py         # ORM models (Telemetry, Anomaly)
├── schemas.py        # Pydantic validation
├── ml_services.py    # Anomaly Detection + Holt-Winters Forecasting
├── dashboard.py      # Streamlit 3D command center
├── edge_simulator.py # Standalone simulator (for local development)
├── .env              # Environment configuration
├── .env.example      # Configuration template
└── requirements.txt  # Dependencies
```

## Cloud Simulator Features

The bundled `CloudEdgeSimulator` provides:

### Realistic AMD Ryzen AI Metrics

| AI Mode | CPU Range | Latency Range |
|---------|-----------|---------------|
| ECO_INT8 | 15-30% | 8-15ms |
| BALANCED_FP16 | 35-55% | 15-25ms |
| PERFORMANCE_FP32 | 60-85% | 25-45ms |

### Demo Spike Feature

Every 15 cycles, the simulator forces a high occupancy spike (>45) to guarantee:
- PERFORMANCE_FP32 mode activation
- Anomaly detection trigger
- Perfect for live demos and judge evaluations

### No psutil Dependency

Cloud servers don't have AMD Ryzen NPUs. The simulator generates authentic-looking metrics without reading actual hardware, making it perfect for cloud deployment.

## Deployment Platforms

### Render.com

```yaml
# render.yaml
services:
  - type: web
    name: aerotwin-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        value: sqlite:///./aerotwin_prod.db
      - key: SIMULATOR_ENABLED
        value: true
```

### Railway

```bash
railway up
```

### Heroku

```bash
heroku create aerotwin-edge
git push heroku main
```

## Local Development with External Simulator

For local testing with the standalone simulator:

```bash
# Terminal 1: Backend (disable bundled simulator)
SIMULATOR_ENABLED=false python main.py

# Terminal 2: External simulator with real psutil metrics
python edge_simulator.py --interval 5

# Terminal 3: Dashboard
streamlit run dashboard.py
```
