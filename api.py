from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Bike Rental Demand Prediction API")

class PredictionRequest(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    visibility: float
    solar_radiation: float
    hour: int
    day_of_week: int
    month: int
    is_holiday: bool
    functioning_day: bool

class PredictionResponse(BaseModel):
    predicted_demand: float
    confidence_interval: tuple
    model_used: str

# Load trained models (replace with your actual models)
models = {
    'lstm': joblib.load('models/lstm_model.pkl'),
    'random_forest': joblib.load('models/rf_model.pkl'),
    'linear_regression': joblib.load('models/lr_model.pkl')
}

@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    try:
        features = np.array([[
            request.temperature, request.humidity, request.wind_speed,
            request.visibility, request.solar_radiation,
            np.sin(2 * np.pi * request.hour / 24),
            np.cos(2 * np.pi * request.hour / 24),
            request.day_of_week, request.month,
            int(request.is_holiday), int(request.functioning_day)
        ]])
        
        predictions = []
        for model_name, model in models.items():
            pred = model.predict(features)[0]
            predictions.append(pred)
        
        final_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)
        
        confidence_interval = (
            final_prediction - 1.96 * std_prediction,
            final_prediction + 1.96 * std_prediction
        )
        
        return PredictionResponse(
            predicted_demand=float(final_prediction),
            confidence_interval=confidence_interval,
            model_used="ensemble"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": len(models)}
