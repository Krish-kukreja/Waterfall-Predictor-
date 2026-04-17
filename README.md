# Waterfall Predictor (DWLR Analysis Pipeline)

Welcome to the **Waterfall Predictor**! This repository houses an end-to-end AI agentic pipeline to forecast water levels, analyze hydrologic trends, and detect anomalies in temporal datasets automatically. It's designed specifically for the **DWLR** (Deep/Digital Water Level Research) initiative.

## Key Features

* **Advanced Forecasting Engine**: Leverages Facebook's `Prophet` alongside an `LSTM` deep learning model to forecast short and long-term water levels based on multi-dimensional context.
* **Intelligent Anomaly Detection**: Uses `Isolation Forest` and `Z-Score` transformations to instantly identify anomalous temporal activities or structural outliers.
* **Auto-Generated Reporting**: Automatically outputs scaled anomaly snapshots and interactive charts to a dedicated visualizations directory (`dwlr_visuals_output/`).
* **Synthetic Dataset Integrated**: A verified `dwlr_synthetic_dataset.csv` is seamlessly packaged alongside the tool, providing an immediate test-bed for running the pipelines right out of the box.

## Project Structure

* `waterfall_predictor.py`: The primary orchestration script encompassing data feature processing, predictive forecasting, and visual intelligence pipelines.
* `dwlr_synthetic_dataset.csv`: The underlying synthetic data table simulating contextual factors like water level and rainfall over time.
* `dwlr_visuals_output/`: Contains auto-exported timeline anomaly models (`.png`, `.html`).
* `requirements.txt`: Lightweight pipeline dependencies.

## How to Run

1. **Install Requirements**: Ensure you have the required python packages available in your environment.
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the Predictor Pipeline**:
   ```bash
   python waterfall_predictor.py
   ```

3. **Analyze the Results**: Standard outputs will stream directly to your CLI reporting regression performance (MAE/RMSE) and identifying any actionable anomalies. Data visualizations will export successfully in the localized outputs directory.
