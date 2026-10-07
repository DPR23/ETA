from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

app = FastAPI(title="NewCold ETA Prediction API")

try:
    model = joblib.load('lgbm_eta_model.pkl')
except Exception as e:
    model = None

class DeliveryFeature(BaseModel):
    distance_km: float
    traffic_index: int
    weather_condition: int
    truck_type: int

@app.post("/predict_eta")
def predict_eta(data: DeliveryFeature):
    if not model:
        raise HTTPException(status_code=500, detail="Model not found. Run train.py first.")
    
    features = [[data.distance_km, data.traffic_index, data.weather_condition, data.truck_type]]
    prediction = model.predict(features)
    return {
        "status": "success",
        "predicted_eta_minutes": round(prediction[0], 2)
    }
