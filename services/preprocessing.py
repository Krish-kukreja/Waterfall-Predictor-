import pandas as pd
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date')
    return df

# I have only interpolated small gaps, its better to leave long gaps as NaN ( better than fake data)
def handle_missing_values(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include = [np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method = 'linear', limit = limit)
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    conditions = (
        (df['Water_Level_m'].between(0,200)) &
        (df['Temperature_C'].between(-5,60)) &
        (df['Rainfall_mm'].between(0,500)) &
        (df['pH'].between(0,14)) &
        (df['Dissolved_Oxygen_mg_L'].between(0,20))
    )
    return df[conditions]

def resample_data(df: pd.DataFrame,freq: str = 'D') -> pd.DataFrame:
    df = df.set_index('Date').resample(freq).mean().reset_index()
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["WaterLevel_rolling7"] = df['Water_Level_m'].rolling(7, min_periods=1).mean()
    df['month'] = df['Date'].dt.month
    df['is_monsoon'] = df['month'].isin([6,7,8,9]).astype(int)
    return df

def normalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    numeric_cols = df.select_dtypes(include = [np.number]).columns
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, scaler


def preprocess_pipeline(path: str, freq: str = 'D',normalize_data: bool = False):
    df = load_data(path)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = resample_data(df, freq = freq)
    df = add_features(df)

    if normalize_data:
        df, _ = normalize(df)
    return df

if __name__ == "__main__":
    # Example usage
    sample_path = "D:\AI ML projects\dwlr_prototype\data\DWLR_Dataset_2023.csv"
    processed_df = preprocess_pipeline(sample_path, normalize_data=False)
    print(processed_df.head())
    print(processed_df.info())
