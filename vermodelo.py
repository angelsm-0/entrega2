import joblib

# cargar modelo
model = joblib.load("mejor_modelo_obesidad.joblib")

print("TIPO DE MODELO:")
print(type(model))

print("\nCOLUMNAS DEL MODELO:")
try:
    print(model.feature_names_in_)
except:
    print("El modelo no tiene feature_names_in_")

print("\nCLASES DEL MODELO:")
try:
    print(model.classes_)
except:
    print("No tiene clases visibles")