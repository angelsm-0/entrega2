from fastapi import FastAPI
from schemas import UserInput
from model_loader import load_model
import numpy as np

app = FastAPI(title="Malnutricion API")

model = load_model()

@app.get("/")
def home():
   return {"message": "API funcionando correctamente"}

@app.post("/predict")
def predict(data: UserInput):

   if model is None:
       return {"error": "Modelo no cargado"}

   input_array = np.array([[
       data.edad,
       data.altura,
       data.peso,
       data.faf,
       data.ch2o,
       data.fcvc
   ]])

   prediction = model.predict(input_array)[0]

   return {
       "prediction": str(prediction)
   }
