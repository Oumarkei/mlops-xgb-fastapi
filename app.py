from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title= "Classifier API",
    description= "This is an XGBoost-based binary classifier API",
    version= "1.0.0"
)

# Let's load our inference pipeline
try:
    pipeline = joblib.load("best_pipeline.pkl")
except Exception as e:
    pipeline = None
    print(f"Erreur de chargement du modèle: {e}")

class PredictionRequest(BaseModel):
    mean_radius : float = Field(description = "mean radius", default = 14.13)
    mean_texture: float= Field(description = "mean texture", default = 19.3)
    mean_perimeter: float= Field(description = "mean perimeter", default = 92)
    mean_area: float= Field(description = "mean area", default = 654.89)
    mean_smoothness: float= Field(description = "mean smoothness", default = 0.01)
    mean_compactness: float= Field(description = "mean compactness", default = 0.10)
    mean_concavity: float= Field(description = "mean concavity", default = 0.09)

# define a get endpoint
@app.get("/")
def sanity_check():
    return {"status": "active API", "pipeline_loaded": pipeline is not None}

@app.post("/predict")
def predict(data: PredictionRequest):
    # Let's check the pipeline is well loaded
    if pipeline is None :
        raise HTTPException(status_code=500, detail= "The inference pipeline is not available")

    input = pd.DataFrame([
        {
            "mean radius": data.mean_radius,
            "mean texture": data.mean_texture,
            "mean perimeter": data.mean_perimeter,
            "mean area": data.mean_area,
            "mean smoothness": data.mean_smoothness,
            "mean compactness": data.mean_compactness,
            "mean concavity": data.mean_concavity,
        }
    ])

    # Let's run inference
    prediction = pipeline.predict(input)[0]
    probability = pipeline.predict_proba(input)[0].max()

    return {
        "prediction": int(prediction),
        "class_name": "Malignant" if prediction == 0 else "Benign",
        "confidence": float(probability)
    }

