# AI-Driven Energy Efficiency in Large Organisations

## Overview
This project implements an AI-powered real-time energy efficiency dashboard for large organizations, particularly focusing on data center operations. It leverages machine learning models to monitor, predict, and optimize energy consumption in virtualized environments, helping reduce operational costs and environmental impact.

## Features
- **Real-Time Monitoring**: Live dashboard displaying current DC power consumption, VM CPU usage, core counts, and estimated power.
- **Anomaly Detection**: Identifies unusual energy consumption patterns using machine learning.
- **Power Forecasting**: Predicts future power usage based on historical data.
- **VM Clustering**: Groups VMs into clusters for better resource management.
- **Recommendations**: Provides actionable insights for VM scaling, downsizing, or shutdown based on usage thresholds.
- **Interactive Web Interface**: Modern, responsive dashboard built with HTML, CSS, and JavaScript.

## Architecture
- **Frontend**: Static HTML/CSS/JS dashboard (`homepage.html`) that fetches data via API calls.
- **Backend**: FastAPI-based REST API (`api/app.py`) serving real-time data and ML inferences.
- **Models**: Pre-trained ML models stored in `api/models/` (forecasting, anomaly detection, clustering, scaling).
- **Data**: Time-series data for data centers and VMs stored in `api/data/` (Parquet format).

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai-driven-energy-efficiency
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```
   pip install fastapi uvicorn joblib pandas numpy scikit-learn pyarrow
   ```

4. Ensure data files are in place:
   - `api/data/datacenter_timeseries.parquet`
   - `api/data/vm_features.parquet`
   - `api/data/vm_level_data.parquet`

5. Run the API server:
   ```
   cd api
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

6. Open `homepage.html` in a web browser to view the dashboard.

## Usage
- Start the FastAPI server as described above.
- Open `homepage.html` in your browser.
- The dashboard will automatically update every 5 seconds with simulated real-time data.
- Metrics include:
  - Current DC Power (W)
  - Anomaly Detection (Yes/No)
  - Predicted Power (W)
  - VM CPU Usage (%)
  - VM Cores
  - VM Estimated Power (W)
  - Cluster ID
  - Recommendation

## API Endpoints
- `GET /health`: Health check.
- `GET /realtime/power`: Returns current DC power and anomaly status.
- `GET /realtime/predict`: Returns predicted power consumption.
- `GET /realtime/vm`: Returns VM metrics, cluster, and recommendation.

## Models
- **Forecast Model**: Predicts power usage (likely a regression model).
- **Anomaly Model**: Detects anomalies (e.g., Isolation Forest).
- **K-Means Model**: Clusters VMs based on features.
- **Scaler**: Standardizes input features for ML models.

Models are pre-trained and loaded from `api/models/`.

## Data
- `datacenter_timeseries.parquet`: Historical DC power data.
- `vm_features.parquet`: VM feature data.
- `vm_level_data.parquet`: Detailed VM-level metrics.

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

## License
This project is licensed under the MIT License. See LICENSE file for details.

## Future Enhancements
- Integrate with real cloud providers (Azure, AWS) for live data.
- Add user authentication and multi-tenant support.
- Implement advanced ML models (e.g., LSTM for forecasting).
- Add alerting and notification systems.
