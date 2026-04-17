import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

class DataFeatureAgent:
    def process(self, raw_data_path):
        data = pd.read_csv(raw_data_path, parse_dates=['date'])

        # Rolling averages and differences
        data['level_ma_3'] = data['water_level'].rolling(3).mean().bfill()
        data['rainfall_ma_7'] = data['rainfall'].rolling(7).mean().bfill()
        data['level_diff'] = data['water_level'].diff().fillna(0)

        # ENHANCEMENT: Cyclical Features for Time / Seasonality
        data['day_of_year'] = data['date'].dt.dayofyear
        data['month'] = data['date'].dt.month
        
        # Sine & Cosine transformations (enables neural network to interpret cyclical nature of time)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

        # Features for modelling (Added cyclical elements)
        feature_cols = [
            'water_level', 'rainfall', 'temp', 'population', 'land_use',
            'level_ma_3', 'rainfall_ma_7', 'level_diff',
            'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]

        # Normalization
        scaler = MinMaxScaler()
        data[feature_cols] = scaler.fit_transform(data[feature_cols])

        return data, feature_cols, scaler


class PredictionAnomalyAgent:
    def run(self, data, feature_cols):
        # ----- Prophet Forecast -----
        df_prophet = data[['date', 'water_level']].rename(
            columns={'date': 'ds', 'water_level': 'y'}
        )
        model_prophet = Prophet(daily_seasonality=True)
        model_prophet.fit(df_prophet)
        future = model_prophet.make_future_dataframe(periods=30)
        forecast_prophet = model_prophet.predict(future)

        # ----- LSTM Forecast (Enhanced with Train/Test Split) -----
        seq_len = 10  
        features = data[feature_cols].values

        X, y = [], []
        for i in range(len(features) - seq_len):
            X.append(features[i:i + seq_len])
            # Assuming 'water_level' is at index 0 of feature_cols
            y.append(features[i + seq_len, 0]) 
            
        X, y = np.array(X), np.array(y)
        
        # ENHANCEMENT: Temporal Train / Test split (80% train, 20% test without shuffling for time series integrity)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # ENHANCEMENT: Deep Learning Architecture Improvement
        lstm_model = Sequential([
            LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2), # Prevents overfitting to the training set
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')

        # ENHANCEMENT: Early Stopping leveraging the correct validation split
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        # ENHANCEMENT: validation_split dynamically tracks optimization metrics realistically
        lstm_model.fit(
            X_train, y_train,
            epochs=30, batch_size=16, verbose=0,
            validation_split=0.15,
            callbacks=[early_stopping]
        )

        # Generate Predictions 
        train_pred = lstm_model.predict(X_train, verbose=0)
        test_pred = lstm_model.predict(X_test, verbose=0)
        
        # Calculate Regression Metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        metrics = {
            'train_mae': train_mae, 'train_rmse': train_rmse,
            'test_mae': test_mae, 'test_rmse': test_rmse
        }
        
        # Rejoin sets across the timeline
        lstm_pred_full = np.concatenate([train_pred, test_pred], axis=0).flatten()

        data_lstm_pred = data.iloc[seq_len:].copy()
        data_lstm_pred['lstm_pred'] = lstm_pred_full
        
        # Capture timeline split for visualizations
        test_start_date = data_lstm_pred.iloc[split_idx]['date']

        # ----- Anomaly Detection (Enhanced with Reconstruction Error) -----
        # Instead of generic stats, anomalies are dynamically assigned where the 
        # Deep Learning model cannot accurately reconstruct the timeline (MSE)
        errors = np.abs(data_lstm_pred['water_level'].values - lstm_pred_full)
        
        # We calculate our threshold boundaries explicitly from the Training set errors
        # Threshold is defined as Statistical Mean Error + (2 * Standard Deviation)
        train_errors = errors[:split_idx]
        threshold = np.mean(train_errors) + 2 * np.std(train_errors)
        
        data_lstm_pred['reconstruction_error'] = errors
        # Flag instances where error is above calculated boundaries (0/1)
        data_lstm_pred['anomaly'] = (data_lstm_pred['reconstruction_error'] > threshold).astype(int)

        return forecast_prophet, data_lstm_pred, test_start_date, metrics


class VisualizationDecisionAgent:
    def create(self, forecast_prophet, data_lstm_pred, test_start_date, metrics):
        # ENHANCEMENT: Non-Blocking Architecture (Export files to folder instead of hanging runtime)
        output_dir = "dwlr_visuals_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Prophet Plot
        fig = px.line(forecast_prophet, x='ds', y='yhat', title='Prophet Forecasted Water Level')
        prophet_path = os.path.join(output_dir, 'prophet_forecast.html')
        fig.write_html(prophet_path)

        # 2. LSTM Timeline & Anomalies Plot
        plt.figure(figsize=(14, 6))
        
        plt.plot(data_lstm_pred['date'], data_lstm_pred['lstm_pred'], label='LSTM Prediction', color='blue', alpha=0.7)
        plt.plot(data_lstm_pred['date'], data_lstm_pred['water_level'], label='Actual Water Level', color='orange', alpha=0.7)
        
        # Mark timeline boundary line (Train vs Unseen Data)
        plt.axvline(x=test_start_date, color='green', linestyle='--', label='Train/Test Split')
        
        # Extract and scatter Anomalies 
        anomalies = data_lstm_pred[data_lstm_pred['anomaly'] == 1]
        plt.scatter(anomalies['date'], anomalies['water_level'], color='red', label='Anomaly (High Reconstruction Error)', zorder=5)
        
        plt.xlabel('Date')
        plt.ylabel('Normalized Water Level')
        plt.title('AI-Augmented DWLR Forecasting & Anomaly Map')
        plt.legend()
        
        lstm_path = os.path.join(output_dir, 'lstm_timeline_anomalies.png')
        plt.savefig(lstm_path, bbox_inches='tight')
        plt.close() 

        # Build Insights Report
        num_anomalies_train = anomalies[anomalies['date'] < test_start_date].shape[0]
        num_anomalies_test = anomalies[anomalies['date'] >= test_start_date].shape[0]
        recent_trend = 'increasing' if data_lstm_pred['water_level'].iloc[-1] > data_lstm_pred['water_level'].iloc[-10] else 'decreasing'
        
        recommendation = (
            f"\n--------------------------------------------------------------\n"
            f"--- AI RECOMMENDATION REPORT ---\n"
            f"--------------------------------------------------------------\n"
            f"Model Regression Performance (Scaled 0-1):\n"
            f"  Train Set -> MAE: {metrics['train_mae']:.4f} | RMSE: {metrics['train_rmse']:.4f}\n"
            f"  Test Set  -> MAE: {metrics['test_mae']:.4f} | RMSE: {metrics['test_rmse']:.4f}\n"
            f"Test Model Interface Deployment Time: {test_start_date.strftime('%Y-%m-%d')}\n"
            f"Detected Anomalies (Training Set Context): {num_anomalies_train}\n"
            f"Detected Anomalies (Unseen Test Set Validation Space): {num_anomalies_test}\n"
            f"Water Level 10-day Recent Action Curve is {recent_trend}.\n\n"
            f"[ACTIONABLE INTELLIGENCE]\n"
            f"Please review LSTM reconstruction anomalies on timeline marked in red.\n"
            f"Visualizations exported directly to '{output_dir}/' directory seamlessly.\n"
            f"--------------------------------------------------------------"
        )
        print(recommendation)
        return recommendation


class DWLREnhancedSystem:
    def __init__(self):
        self.data_agent = DataFeatureAgent()
        self.pred_anomaly_agent = PredictionAnomalyAgent()
        self.visual_decision_agent = VisualizationDecisionAgent()

    def run_pipeline(self, raw_data_path):
        print("-> [1/3] Extracting features & initializing Data Pipelines...")
        data, feature_cols, scaler = self.data_agent.process(raw_data_path)
        
        print("-> [2/3] Executing Prophet Framework & Deep Learning LSTM splits...")
        forecast_prophet, data_lstm_pred, test_start_date, metrics = self.pred_anomaly_agent.run(data, feature_cols)
        
        print("-> [3/3] Engineering Anomaly Visuals securely tracking results...")
        recommendation = self.visual_decision_agent.create(forecast_prophet, data_lstm_pred, test_start_date, metrics)
        return recommendation

if __name__ == "__main__":
    dataset_path = "dwlr_synthetic_dataset.csv"
    if os.path.exists(dataset_path):
        system = DWLREnhancedSystem()
        system.run_pipeline(dataset_path)
    else:
        print(f"Error: Could not locate operational system datasets under `{dataset_path}`")
