import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


class DataFeatureAgent:
    def process(self, raw_data_path):
        """
        Here I read the raw data, created some extra features,
        and scaled everything so the models wouldn’t get confused by different units.
        """

        # I loaded the dataset and made sure the 'date' column was parsed as datetime
        data = pd.read_csv(raw_data_path, parse_dates=['date'])

        # I calculated a 3-day rolling average of water level
        # to smooth short-term fluctuations
        data['level_ma_3'] = data['water_level'].rolling(3).mean().fillna(method='bfill')

        # I also calculated a 7-day rolling average of rainfall
        # since rainfall usually follows weekly patterns
        data['rainfall_ma_7'] = data['rainfall'].rolling(7).mean().fillna(method='bfill')

        # I took the difference in water level from the previous day
        # so I could capture whether it was rising or falling
        data['level_diff'] = data['water_level'].diff().fillna(0)

        # These were the features I decided to use for the models
        feature_cols = [
            'water_level', 'rainfall', 'temp', 'population', 'land_use',
            'level_ma_3', 'rainfall_ma_7', 'level_diff'
        ]

        # I normalized all of them between 0 and 1
        # so that no single feature dominated the others
        scaler = MinMaxScaler()
        data[feature_cols] = scaler.fit_transform(data[feature_cols])

        return data, feature_cols

class PredictionAnomalyAgent:
    def run(self, data, feature_cols):
        """
        I used Prophet and LSTM to forecast water levels,
        and I applied Isolation Forest and Z-score methods to detect anomalies.
        """

        # ----- Prophet Forecast -----
        # Prophet needed 'ds' for date and 'y' for target, so I renamed the columns
        df_prophet = data[['date', 'water_level']].rename(
            columns={'date': 'ds', 'water_level': 'y'}
        )

        # I set up Prophet with daily seasonality and trained it
        model_prophet = Prophet(daily_seasonality=True)
        model_prophet.fit(df_prophet)

        # I asked it to predict 30 days into the future
        future = model_prophet.make_future_dataframe(periods=30)
        forecast_prophet = model_prophet.predict(future)


        # ----- LSTM Forecast -----
        # For LSTM, I used the past 10 days to predict the next day
        seq_len = 10  
        features = data[feature_cols].values

        # I built sequences (X) and labels (y)
        X, y = [], []
        for i in range(len(features) - seq_len):
            X.append(features[i:i + seq_len])       # past 10 days
            y.append(features[i + seq_len, 0])      # next day's water level

        X, y = np.array(X), np.array(y)

        # I built a simple LSTM model with one hidden layer and an output layer
        lstm_model = Sequential([
            LSTM(50, input_shape=(X.shape[1], X.shape[2])),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')

        # I trained it for up to 20 epochs with early stopping
        lstm_model.fit(
            X, y,
            epochs=20, batch_size=16, verbose=0,
            callbacks=[EarlyStopping(patience=3)]
        )

        # Then I generated predictions
        lstm_pred = lstm_model.predict(X)

        # I aligned the predictions with the correct dates
        data_lstm_pred = data[seq_len:].copy()
        data_lstm_pred['lstm_pred'] = lstm_pred.flatten()


        # ----- Anomaly Detection -----
        # First, I applied Isolation Forest to find unusual water levels
        iso = IsolationForest(contamination=0.05, random_state=42)
        anomalies_iso = iso.fit_predict(data[['water_level', 'level_ma_3']])
        # (-1 meant anomaly, 1 meant normal)

        # Then I used the Z-score method
        # I calculated a rolling mean and std deviation
        rolling_mean = data['water_level'].rolling(10).mean()
        rolling_std = data['water_level'].rolling(10).std()
        z_score = (data['water_level'] - rolling_mean) / rolling_std

        # Any point more than 2 std deviations away, I marked as anomaly
        anomalies_z = (np.abs(z_score) > 2).astype(int)

        # If either method flagged a point, I marked it as anomaly=1
        data['anomaly'] = np.where((anomalies_iso == -1) | (anomalies_z == 1), 1, 0)

        return forecast_prophet, data, data_lstm_pred

class VisualizationDecisionAgent:
    def create(self, forecast_prophet, anomaly_data, lstm_data):
        # Prophet plot (interactive)
        fig = px.line(forecast_prophet, x='ds', y='yhat', title='Prophet Forecasted Water Level')
        fig.show()

        # LSTM + Anomaly plot (fixed alignment)
        plt.figure(figsize=(12,5))
        
        # ✅ use lstm_data for aligned predictions
        plt.plot(lstm_data['date'], lstm_data['lstm_pred'], label='LSTM Prediction', color='blue')
        
        # Actual water levels
        plt.plot(anomaly_data['date'], anomaly_data['water_level'], label='Actual Water Level', color='orange')
        
        # Anomalies (scatter)
        plt.scatter(
            anomaly_data.loc[anomaly_data['anomaly'] == 1, 'date'],
            anomaly_data.loc[anomaly_data['anomaly'] == 1, 'water_level'],
            color='red', label='Anomaly'
        )
        
        plt.xlabel('Date')
        plt.ylabel('Normalized Water Level')
        plt.title('LSTM Prediction & Anomalies')
        plt.legend()
        plt.show()

        # Recommendations
        num_anomalies = anomaly_data['anomaly'].sum()
        trend = 'increasing' if anomaly_data['water_level'].iloc[-1] > anomaly_data['water_level'].iloc[0] else 'decreasing'
        recommendation = f"Detected {num_anomalies} anomalous days. Water level trend is {trend}. Adjust water usage and monitor recharge."
        print(recommendation)
        return recommendation

# -----------------------------
# Orchestrator
# -----------------------------
class DWLRAgenticSystem:
    def __init__(self):
        self.data_agent = DataFeatureAgent()
        self.pred_anomaly_agent = PredictionAnomalyAgent()
        self.visual_decision_agent = VisualizationDecisionAgent()

    def run_pipeline(self, raw_data_path):
        data, feature_cols = self.data_agent.process(raw_data_path)
        forecast_prophet, anomaly_data, lstm_data = self.pred_anomaly_agent.run(data, feature_cols)
        recommendation = self.visual_decision_agent.create(forecast_prophet, anomaly_data, lstm_data)
        return recommendation

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    dataset_path = "dwlr_synthetic_dataset.csv"  # make sure CSV is in same folder
    system = DWLRAgenticSystem()
    system.run_pipeline(dataset_path)
