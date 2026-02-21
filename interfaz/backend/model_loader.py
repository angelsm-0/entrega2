import os
MODEL_PATH = os.path.join("model", "model.pkl")

def load_model():
   if os.path.exists(MODEL_PATH):
       return joblib.load(MODEL_PATH)
   else:
       return None
