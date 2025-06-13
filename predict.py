import joblib
import numpy as np
from tensorflow.keras.models import load_model

class BikeDemandPredictor:
    def __init__(self):
        self.lstm_model = load_model("models/lstm_model.h5")
        self.rf_model = joblib.load("models/rf_model.pkl")
    
    def predict(self, input_data):
        lstm_pred = self.lstm_model.predict(input_data)
        rf_pred = self.rf_model.predict(input_data)
        return (lstm_pred + rf_pred) / 2  # Ensemble prediction

# Example usage
if __name__ == "__main__":
    predictor = BikeDemandPredictor()
    sample_input = np.array([[25, 70, 0.5, 0.8]])  # Temp, Humidity, Hour_sin, Hour_cos
    print("Predicted Demand:", predictor.predict(sample_input))
