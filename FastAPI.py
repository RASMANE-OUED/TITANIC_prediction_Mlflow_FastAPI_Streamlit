from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Titanic Survival Prediction API")

# Chemins vers artefacts sauvegardés
MODEL_PATH = "artifacts/best_model.joblib"
SCALER_PATH = "artifacts/scaler.joblib"
LE_SEX_PATH = "artifacts/label_encoder_sex.joblib"
LE_EMBARKED_PATH = "artifacts/label_encoder_embarked.joblib"
FEATURE_NAMES_PATH = "artifacts/feature_names.joblib"

# Chargement des artefacts au démarrage
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le_sex = joblib.load(LE_SEX_PATH)
le_embarked = joblib.load(LE_EMBARKED_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

class Passenger(BaseModel):
    pclass: int = Field(..., ge=1, le=3)
    sex: str = Field(..., pattern="^(male|female)$")
    age: float = Field(..., ge=0)
    sibsp: int = Field(..., ge=0)
    parch: int = Field(..., ge=0)
    fare: float = Field(..., ge=0)
    embarked: str = Field(..., pattern="^[SCQ]$")

def _safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        df = (df - df.min()) / (df.max() - df.min() + 1e-9)
        return df
    return model.predict(X)

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API Titanic"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(passenger: Passenger):
    # Encoder et préparer les données en entrée
    row = {
        "pclass": passenger.pclass,
        "sex": le_sex.transform([passenger.sex])[0],
        "age": passenger.age,
        "sibsp": passenger.sibsp,
        "parch": passenger.parch,
        "fare": passenger.fare,
        "embarked": le_embarked.transform([passenger.embarked])[0],
    }
    X = pd.DataFrame([row], columns=feature_names)
    X_scaled = scaler.transform(X)
    proba = _safe_predict_proba(model, X_scaled)
    prob_survie = float(proba[0]) if np.ndim(proba) > 0 else float(proba)

    prediction = 1 if prob_survie >= 0.5 else 0
    return {"prediction": prediction, "probability_survival": prob_survie}


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import uvicorn

    uvicorn.run("fastAPI:app", host="127.0.0.1", port=8000, reload=True)
