import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import joblib

# 1. Cargar y preparar datos
url = 'https://raw.githubusercontent.com/pymche/Machine-Learning-Obesity-Classification/master/ObesityDataSet_raw_and_data_sinthetic.csv'
data_path = 'datos/Malnutricion.csv'

# Asegurar que el directorio de datos existe
if not os.path.exists('datos'):
    os.makedirs('datos')

# Descargar si no existe
if not os.path.exists(data_path):
    print("Descargando dataset de obesidad...")
    try:
        df = pd.read_csv(url)
        df.to_csv(data_path, index=False)
        print("Dataset guardado en 'datos/Malnutricion.csv'")
    except Exception as e:
        print(f"Error al descargar: {e}")
        # Intentamos recrear una muestra pequeña si falla la descarga o usamos datos sintéticos
        exit(1)
else:
    print("Usando dataset local 'datos/Malnutricion.csv'")
    df = pd.read_csv(data_path)

# 2. Preprocesamiento básico
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Codificar el target (NObeyesdad tiene 7 niveles)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_

# Identificar columnas por tipo
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Definir transformadores para el Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# División de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Configurar MLFlow
# Usaremos SQLite para el backend local si es posible, o simplemente el filesystem por defecto
mlflow.set_experiment("Modelos de Clasificación de Obesidad")

def train_and_log_model(model_name, model):
    print(f"\nEntrenando modelo: {model_name}...")
    with mlflow.start_run(run_name=model_name):
        # Crear pipeline completo para incluir el preprocesamiento en el modelo guardado
        clf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Entrenar el pipeline
        clf_pipeline.fit(X_train, y_train)
        
        # Realizar predicciones
        y_pred = clf_pipeline.predict(X_test)
        
        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Registrar parámetros (si existen)
        params = model.get_params()
        # Solo logueamos algunos parámetros clave para no saturar
        key_params = ['n_estimators', 'max_depth', 'learning_rate', 'criterion']
        for p in key_params:
            if p in params and params[p] is not None:
                mlflow.log_param(f"{model_name}_{p}", params[p])
        
        # Registrar métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Registrar el modelo
        mlflow.sklearn.log_model(clf_pipeline, artifact_path="model", registered_model_name=model_name)
        
        # Guardar clases del LabelEncoder como un artefacto extra (opcional)
        with open("target_classes.txt", "w") as f:
            for cls in target_names:
                f.write(f"{cls}\n")
        mlflow.log_artifact("target_classes.txt")
        
        print(f"Completado: {model_name}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return {"accuracy": acc, "f1_weighted": f1}

# 4. Comparar diferentes algoritmos
models_to_test = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)),
    ("XGBoost", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)),
    ("SVM", SVC(probability=True, random_state=42)),
    ("KNN", KNeighborsClassifier(n_neighbors=5))
]

results = []

for name, model in models_to_test:
    try:
        metrics = train_and_log_model(name, model)
        results.append({
            "Modelo": name,
            "Accuracy": metrics["accuracy"],
            "F1-Score": metrics["f1_weighted"]
        })
    except Exception as e:
        print(f"Error entrenando {name}: {e}")

# 5. Mostrar Tabla Comparativa e Identificar el Mejor
print("\n" + "="*55)
print(f"{'Modelo':<20} | {'Accuracy':<12} | {'F1-Score':<10}")
print("-" * 55)
for res in sorted(results, key=lambda x: x['Accuracy'], reverse=True):
    print(f"{res['Modelo']:<20} | {res['Accuracy']:<12.4f} | {res['F1-Score']:<10.4f}")
print("="*55)

# 6. Guardar y Registrar el Mejor Modelo
if results:
    best_result = max(results, key=lambda x: x['Accuracy'])
    best_model_name = best_result["Modelo"]
    
    print(f"\n🏆 EL MEJOR MODELO ES: {best_model_name} con Accuracy: {best_result['Accuracy']:.4f}")
    
    # Buscar el objeto del modelo ganador para guardarlo localmente
    for name, model_obj in models_to_test:
        if name == best_model_name:
            # Crear el pipeline final para el mejor modelo
            best_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model_obj)
            ])
            # Re-entrenar el mejor pipeline
            best_pipeline.fit(X_train, y_train)
            
            # Guardar localmente
            output_file = "mejor_modelo_obesidad.joblib"
            joblib.dump(best_pipeline, output_file)
            print(f"✅ Mejor modelo guardado localmente como: {output_file}")
            
            # Registrar en MLflow con un nombre especial
            with mlflow.start_run(run_name="Seleccion_Mejor_Modelo"):
                mlflow.log_param("best_model_type", best_model_name)
                mlflow.log_metric("best_accuracy", best_result["Accuracy"])
                mlflow.sklearn.log_model(best_pipeline, "model")
                print(f"✅ Mejor modelo registrado en MLflow en la corrida 'Seleccion_Mejor_Modelo'")
            break

print("\n--- PROCESO FINALIZADO ---")
print("Para visualizar los resultados detallados en MLflow, ejecute:")
print("mlflow ui --host 0.0.0.0 --port 8050 --allowed-hosts \"*\"")




