from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="SensorGuard API",
    description="API for predictive anomaly detection on turbofan engines.",
    version="1.0.0"
)

# Load models at startup
MODEL_PATH = "saved_models/random_forest_model.pkl"
SCALER_PATH = "saved_models/scaler.pkl"

model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and Scaler loaded successfully.")
    else:
        print("Warning: Model or Scaler not found. Run train.py first.")

class PredictionRequest(BaseModel):
    # Expecting 84 engineered features as described in the ML Pipeline
    features: list[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if len(request.features) != 84:
        raise HTTPException(status_code=400, detail=f"Expected 84 features, got {len(request.features)}")
    
    try:
        # Reshape and scale
        X = np.array(request.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        # Assuming 0 is Normal, 1 is Failure based on typical RUL labeling
        result = "Failure" if prediction == 1 else "Normal"
        
        return {
            "prediction": int(prediction),
            "status": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
