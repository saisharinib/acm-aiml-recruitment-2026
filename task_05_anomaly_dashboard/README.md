# Task 5: Interactive Anomaly Detection Dashboard (Streamlit)

## Overview
An interactive Streamlit app for exploring anomaly detection on IoT sensor data. The user changes the detection settings in a sidebar and the charts and metrics update in real time. It uses the sensor dataset and the two methods from Task 1: the IQR rule and Isolation Forest.

![Dashboard overview](images/dashboard_overview.png)

## Features
- **Method switch:** show results for IQR or Isolation Forest.
- **Live controls:** IQR multiplier, Isolation Forest contamination, number of trees, and the number of hours shown in the time plot.
- **Live metrics:** readings, flagged count, share flagged and, when labels exist, precision, recall and F1.
- **Time series tab:** each sensor with flagged readings in red.
- **Scatter tab:** pick any two sensors for the axes and optionally circle the true anomalies.
- **Distributions tab:** boxplots whose whiskers follow the IQR multiplier, and the Isolation Forest score histogram.
- **Method comparison tab:** side-by-side metrics and the share of each anomaly type caught.
- **Flagged data tab:** a table of flagged readings with a CSV download.
- **Upload your own CSV** with columns `temperature`, `vibration` and `power`. Add an `is_anomaly` column (0 or 1) to see the evaluation metrics.

## Screenshots
**Contextual anomalies in the scatter view**

![Scatter with true anomalies](images/scatter_truth.png)

**Method comparison**

![Method comparison](images/comparison.png)

**Settings changed (higher IQR multiplier)**

![Sliders changed](images/sliders_changed.png)

## How It Works
1. The data is loaded from `sensor_data.csv` (or from an uploaded file).
2. **IQR:** a reading is flagged if any sensor falls outside `[Q1 - k*IQR, Q3 + k*IQR]`, where `k` is the slider value.
3. **Isolation Forest:** the features are standardised and an `IsolationForest` is fitted with the chosen number of trees and contamination.
4. Streamlit re-runs the script whenever a widget changes, so all results refresh instantly.
5. Isolation Forest results are cached with `st.cache_data`, so unchanged settings do not recompute.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501 in a browser. Keep `sensor_data.csv` in the same folder as `app.py`.

## Project Structure
```
task_05_anomaly_dashboard/
├── app.py
├── sensor_data.csv
├── requirements.txt
├── images/
└── README.md
```

## Limitations and Future Work
- The default data is synthetic, so real sensor data would be messier.
- The dashboard treats each reading independently and ignores time order and the daily cycle.
- Possible additions: more algorithms (Local Outlier Factor, One-Class SVM), a live data stream, and deployment on Streamlit Community Cloud.