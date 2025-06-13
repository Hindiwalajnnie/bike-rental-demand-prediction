import pandas as pd
import numpy as np

class BikeRentalPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def preprocess_data(self, df):
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Engineer weather features
        df = self.engineer_weather_features(df)
        
        # Scale numerical features
        numerical_cols = ['Temperature', 'Humidity', 'Wind_speed', 
                         'Visibility', 'Solar_Radiation']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def create_temporal_features(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
        df['Day_of_week'] = df['Date'].dt.dayofweek
        
        return df
    
    def engineer_weather_features(self, df):
        # Weather comfort index
        df['Weather_Score'] = (
            (df['Temperature'] - df['Temperature'].mean()) / df['Temperature'].std() +
            (df['Humidity'].mean() - df['Humidity']) / df['Humidity'].std() +
            (df['Wind_speed'].mean() - df['Wind_speed']) / df['Wind_speed'].std()
        ) / 3
        
        # Temperature categories
        df['Temp_Category'] = pd.cut(df['Temperature'], 
                                   bins=[-np.inf, 0, 10, 20, 30, np.inf],
                                   labels=['Very_Cold', 'Cold', 'Cool', 'Warm', 'Hot'])
        
        return df
