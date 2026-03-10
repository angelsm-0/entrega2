import joblib


MODEL_PATH = "mejor_modelo_obesidad.joblib"


def load_model():

    model = joblib.load(MODEL_PATH)

    return model