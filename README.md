# Predicción de Niveles de Obesidad

Sistema de **Machine Learning para la clasificación de niveles de obesidad** basado en hábitos alimenticios, estilo de vida y características físicas del usuario.

---

## Descripción

El proyecto integra:

- Entrenamiento y comparación de modelos
- Seguimiento de experimentos con MLflow
- API de inferencia con FastAPI
- Dashboard interactivo con Streamlit

---

## Arquitectura

```
Usuario
   │
   ▼
Dashboard (Streamlit)
   │
   ▼
API de Predicción (FastAPI)
   │
   ▼
Pipeline de Machine Learning (Scikit-learn)
   │
   ▼
Modelo Entrenado (.joblib)
   │
   ▼
MLflow Tracking
```

---

## Tecnologías

| Categoría | Tecnologías |
|-----------|-------------|
| Lenguaje | Python |
| ML | scikit-learn, XGBoost |
| API | FastAPI |
| Interfaz | Streamlit, Plotly |
| Tracking | MLflow |
| Datos | Pandas, NumPy |

---

## Dataset

Se utiliza el **Obesity Dataset**, que contiene información sobre hábitos alimenticios, actividad física, consumo de agua y alcohol, uso de tecnología y características físicas.

### Variables de entrada

| Variable | Descripción |
|----------|-------------|
| `Age` | Edad |
| `Gender` | Género |
| `Height` | Altura |
| `Weight` | Peso |
| `family_history_with_overweight` | Historial familiar de sobrepeso |
| `FAVC` | Consumo frecuente de alimentos calóricos |
| `FCVC` | Frecuencia de consumo de vegetales |
| `NCP` | Número de comidas principales |
| `CAEC` | Consumo de alimentos entre comidas |
| `SMOKE` | Fumador |
| `CH2O` | Consumo de agua diario |
| `SCC` | Monitoreo de calorías consumidas |
| `FAF` | Frecuencia de actividad física |
| `TUE` | Tiempo de uso de dispositivos tecnológicos |
| `CALC` | Consumo de alcohol |
| `MTRANS` | Medio de transporte habitual |

**Variable objetivo:** `NObeyesdad` — nivel de obesidad.

---

## Modelos Evaluados

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- XGBoost

Los experimentos fueron registrados con **MLflow** para comparar métricas y seleccionar el mejor modelo.

---

## Estructura del Proyecto

```
entrega2/
│
├── api/
│   ├── app.py
│   └── schemas.py
│
├── interfaz/
│   └── app.py
│
├── datos/
├── mlruns/
├── modelo_obesidad.py
├── mejor_modelo_obesidad.joblib
├── mlflow.db
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/angelsm-0/entrega2.git

# Entrar al proyecto
cd entrega2

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### 1. Entrenar el modelo

```bash
python modelo_obesidad.py
```

Este proceso entrena múltiples modelos, evalúa métricas, guarda el mejor modelo y registra los experimentos en MLflow.

### 2. Visualizar experimentos (MLflow)

```bash
mlflow ui
```

Abre `http://localhost:5000` para visualizar experimentos, métricas, parámetros y comparaciones entre modelos.

### 3. Ejecutar la API

```bash
uvicorn api.app:app --reload
```

Documentación disponible en `http://127.0.0.1:8000/docs`

**Endpoint:** `POST /predict`

**Request de ejemplo:**

```json
{
  "Gender": "Male",
  "Age": 25,
  "Height": 1.75,
  "Weight": 70,
  "family_history_with_overweight": "yes",
  "FAVC": "yes",
  "FCVC": 2,
  "NCP": 3,
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "CH2O": 2,
  "SCC": "no",
  "FAF": 1,
  "TUE": 1,
  "CALC": "Sometimes",
  "MTRANS": "Public_Transportation"
}
```

**Respuesta de ejemplo:**

```json
{
  "prediction_code": 1,
  "prediction_label": "Normal Weight"
}
```

### 4. Ejecutar el Dashboard

```bash
streamlit run interfaz/frontend/app.py
```

Abre `http://localhost:8501` para ingresar datos, enviarlos a la API, obtener predicciones y visualizar gráficos analíticos.

---

## Resultados

El sistema clasifica a los usuarios en una de las siguientes categorías:

| Código | Categoría |
|--------|-----------|
| 0 | Insufficient Weight |
| 1 | Normal Weight |
| 2 | Overweight Level I |
| 3 | Overweight Level II |
| 4 | Obesity Type I |
| 5 | Obesity Type II |
| 6 | Obesity Type III |

---

## Autor

**Luis Angel S.M.**  

**Miguel Angel S.M.**  

Proyecto desarrollado como parte de la Maestría en Inteligencia Artificial.

---

## Licencia

Este proyecto fue desarrollado con fines académicos.
