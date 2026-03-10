from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from api.schemas import ObesityInput

app = FastAPI(
    title="Obesity Prediction API",
    description="API para predicción de niveles de obesidad",
    version="1.0"
)

MODEL_PATH = "mejor_modelo_obesidad.joblib"

model = joblib.load(MODEL_PATH)

class_labels = [
    "Insufficient Weight",
    "Normal Weight",
    "Overweight Level I",
    "Overweight Level II",
    "Obesity Type I",
    "Obesity Type II",
    "Obesity Type III"
]


@app.get("/")
def home():
    return {
        "service": "Obesity Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "model_loaded": True
    }


@app.post("/predict")
def predict(data: ObesityInput):

    try:

        input_dict = data.dict()

        df = pd.DataFrame([input_dict])

        prediction = model.predict(df)

        prediction_code = int(prediction[0])

        prediction_label = class_labels[prediction_code]

        return {
            "prediction_code": prediction_code,
            "prediction_label": prediction_label
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error en inferencia: {str(e)}"
        )