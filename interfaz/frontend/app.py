import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
# Configuración de la página con estética premium
st.set_page_config(
    page_title="VitaPredict AI | Obesity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Estilos personalizados para mejorar la estética
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)
API_URL = "http://127.0.0.1:8000/predict"
# Título y encabezado
st.title("🧬 VitaPredict AI: Análisis de Obesidad LatAm")
st.markdown("---")
# --- SIDEBAR: INGRESO DE DATOS ---
st.sidebar.header("📥 Entrada de Datos")
st.sidebar.markdown("Complete la información del paciente")
with st.sidebar:
    st.subheader("👤 Demografía")
    edad = st.slider("Edad", 15, 65, 25)
    genero = st.selectbox("Género", ["Femenino", "Masculino"])
    altura = st.slider("Altura (m)", 1.4, 2.2, 1.70, step=0.01)
    peso = st.slider("Peso (kg)", 40, 180, 70, step=0.5)
    
    st.subheader("🥗 Hábitos Alimenticios")
    faf = st.slider("Actividad Física (0-3)", 0, 3, 1)
    ch2o = st.slider("Consumo Agua (L/día)", 1, 3, 2)
    fcvc = st.slider("Consumo Vegetales (1-3)", 1, 3, 2)
    favc = st.selectbox("Comida Alta en Calorías", ["Sí", "No"])
    calc = st.selectbox("Consumo de Alcohol", ["Nunca", "A veces", "Frecuentemente"])
    predict_btn = st.button("🚀 REALIZAR PREDICCIÓN")
# --- MAIN CONTENT ---
if predict_btn:
    # Preparar el payload para la API
    payload = {
        "Age": edad,
        "Height": altura,
        "Weight": peso,
        "FAF": faf,
        "CH2O": ch2o,
        "FCVC": fcvc,
        "Gender": 1 if genero == "Masculino" else 0,
        "FAVC": 1 if favc == "Sí" else 0,
        "CALC": calc
    }
    try:
        # Simulación de llamada a API (o llamada real si está activa)
        # Comentado para evitar errores si la API no está corriendo
        # response = requests.post(API_URL, json=payload, timeout=2)
        
        # Simulación de respuesta para el prototipo
        imc = peso / (altura ** 2)
        
        # Lógica simulada de predicción basada en IMC (el modelo real haría esto en el backend)
        if imc < 18.5: prediction = "Peso Insuficiente"
        elif imc < 25: prediction = "Peso Normal"
        elif imc < 30: prediction = "Sobrepeso"
        else: prediction = "Obesidad"
        # --- SECCIÓN DE RESULTADOS ---
        st.subheader("🎯 Resultado de la Evaluación")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.metric("Nivel Predicho", prediction)
        with col2:
            st.metric("IMC Calculado", f"{imc:.2f}")
        
        with col3:
            # Gauge Chart con Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=imc,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Índice de Masa Corporal", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [10, 45], 'tickwidth': 1},
                    'bar': {'color': "#4f46e5"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [10, 18.5], 'color': '#3b82f6'},
                        {'range': [18.5, 25], 'color': '#10b981'},
                        {'range': [25, 30], 'color': '#f59e0b'},
                        {'range': [30, 45], 'color': '#ef4444'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': imc
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        # --- TABLERO ANALÍTICO ---
        st.markdown("---")
        st.subheader("📊 Exploración de Datos y Correlaciones")
        
        dash_col1, dash_col2 = st.columns(2)
        
        with dash_col1:
            # Gráfico de Importancia (Plotly)
            importancias = {
                'Peso': 0.95, 'FAVC': 0.85, 'FAF': 0.72, 
                'FCVC': 0.61, 'CH2O': 0.45, 'Edad': 0.38
            }
            df_imp = pd.DataFrame(list(importancias.items()), columns=['Variable', 'Importancia'])
            fig_imp = px.bar(df_imp, x='Importancia', y='Variable', orientation='h',
                             title="Importancia de Variables en el Modelo",
                             color='Importancia', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)
        with dash_col2:
            # Scatter Plot Correlación (Plotly)
            np.random.seed(42)
            n_points = 100
            mock_data = pd.DataFrame({
                'Actividad Física': np.random.uniform(0, 3, n_points),
                'IMC': 20 + np.random.normal(5, 5, n_points) + (3 - np.random.uniform(0, 3, n_points)) * 2,
                'Categoria': np.random.choice(['Normal', 'Sobrepeso', 'Obesidad'], n_points)
            })
            fig_corr = px.scatter(mock_data, x='Actividad Física', y='IMC', color='Categoria',
                                 title="Correlación: Actividad Física vs IMC",
                                 trendline="ols")
            st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.error(f"Error al conectar con el servidor: {e}")
        st.info("Asegúrese de que la API de FastAPI esté corriendo en http://127.0.0.1:8000")
else:
    # Estado inicial cuando no hay predicción
    st.info("👋 Bienvenid@. Ingrese los datos en el panel izquierdo y haga clic en 'Realizar Predicción' para comenzar el análisis.")
    
    # Mostrar Dashboard estático de ejemplo
    st.subheader("Vista General del Dataset (Muestra)")
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Gráfico de tarta de ejemplo
        labels = ['Bajo Peso', 'Normal', 'Sobrepeso', 'Obesidad']
        values = [15, 35, 30, 20]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig_pie.update_layout(title_text="Distribución de Categorías en LatAm")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        st.image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", caption="Análisis de Tendencias Regionales (Placeholder)")
