# Manual de Usuario — Tablero de Predicción de Obesidad

> Interfaz interactiva para la exploración de datos y predicción de niveles de obesidad mediante Machine Learning.

---

## Descripción

El tablero permite visualizar información relacionada con el análisis de obesidad utilizando modelos de machine learning. La aplicación facilita la exploración de datos y la obtención de predicciones a partir de variables relacionadas con hábitos alimenticios y estilo de vida.

Desarrollado con **Streamlit** para una interacción sencilla e intuitiva desde el navegador.

---

## Acceso

Una vez ejecutada la aplicación, el tablero estará disponible en:

```
http://localhost:8501
```

Para iniciarlo:

```bash
streamlit run interfaz/app.py
```

---

## Funcionalidades Principales

### Visualización de Datos
Explora estadísticas generales del conjunto de datos, incluyendo distribuciones y variables relevantes para el análisis de obesidad.

### Predicción del Modelo
Ingresa características del usuario para obtener una predicción del nivel de obesidad. Las variables consideradas son:

| Variable | Descripción |
|----------|-------------|
| Edad | Años del usuario |
| Peso | Peso corporal en kg |
| Hábitos alimenticios | Frecuencia y tipo de consumo de alimentos |
| Actividad física | Nivel y frecuencia de ejercicio |
| Consumo de agua | Ingesta diaria de agua |
| Uso de transporte | Medio de transporte habitual |

### Exploración de Resultados
Visualiza los resultados del modelo y comprende cómo los distintos factores influyen en la predicción generada.

---

## Flujo de Uso

```
1. Ingresar al tablero desde el navegador
        │
        ▼
2. Seleccionar o ingresar los valores de las variables
        │
        ▼
3. Ejecutar la predicción
        │
        ▼
4. Analizar el resultado generado por el modelo
```

---

## Público Objetivo

Este tablero está orientado a:

- **Estudiantes** que deseen explorar aplicaciones de ML en salud
- **Investigadores** interesados en factores asociados a la obesidad
- **Analistas de datos** que trabajen con este tipo de modelos
- **Usuarios en general** con interés en comprender cómo sus hábitos influyen en su nivel de salud

---

## Consideraciones Importantes

> **Aviso:** Los resultados generados por el modelo corresponden a **predicciones basadas en datos históricos** y **no deben interpretarse como diagnósticos médicos**. Para cualquier evaluación de salud, consultar a un profesional médico calificado.

---

## Licencia

Este proyecto fue desarrollado con fines académicos.
