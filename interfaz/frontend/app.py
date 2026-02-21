import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
# Configuración de la página
st.set_page_config(
    page_title="Obesity Predictor Dashboard",
    page_icon="🏥",
    layout="wide"
)
# --- ESTILOS CSS PARA REPLICAR EL MOCKUP ---
st.markdown("""
    <style>
    /* Estilo General */
    .main {
        background-color: #f0f4f8;
    }
    
    /* Contenedores tipo Card del Mockup */
    .mockup-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #d1d9e6;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .card-title {
        color: #2c3e50;
        font-weight: bold;
        border-bottom: 1px solid #ebebeb;
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-size: 1.2rem;
    }
    /* Barra de Riesgo (Semáforo) */
    .risk-bar {
        display: flex;
        height: 25px;
        width: 100%;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 10px;
    }
    .segment { flex: 1; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8rem; font-weight: bold; }
    .normal { background-color: #6fb064; }
    .riesgo { background-color: #d4e157; color: #558b2f; }
    .alerta { background-color: #ffb74d; color: #ef6c00; }
    .obesidad { background-color: #c62828; }
    
    /* Botones */
    .stButton>button {
        background-color: #34495e;
        color: white;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
API_URL = "http://127.0.0.1:8000/predict"
# --- SIDEBAR: DATOS DEL USUARIO ---
st.sidebar.markdown('<div class="card-title">Datos del Usuario</div>', unsafe_allow_html=True)
with st.sidebar:
    edad = st.number_input("Edad:", 1, 100, 25)
    genero = st.selectbox("Género:", ["Femenino", "Masculino"])
    altura = st.number_input("Altura (m):", 0.5, 2.5, 1.75, step=0.01)
    peso = st.number_input("Peso (kg):", 10, 300, 70)
    
    st.markdown("---")
    st.subheader("Hábitos y Estilo de Vida")
    favc = st.selectbox("Consumo de Calorías (Altas):", ["no", "yes"])
    fcvc = st.slider("Consumo de Vegetales (1-3):", 1, 3, 2)
    ncp = st.slider("Comidas Principales (1-4):", 1, 4, 3)
    ch2o = st.slider("Consumo de Agua (1-3):", 1, 3, 2)
    caec = st.selectbox("Consumo entre comidas:", ["no", "Sometimes", "Frequently", "Always"])
    calc = st.selectbox("Consumo de alcohol:", ["no", "Sometimes", "Frequently", "Always"])
    
    st.markdown("---")
    faf = st.slider("Actividad Física (0-3):", 0, 3, 1)
    tue = st.slider("Uso de Tecnología (0-2):", 0, 2, 1)
    mtrans = st.selectbox("Medio de Transporte:", ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"])
    scc = st.selectbox("Monitoreo de Calorías:", ["no", "yes"])
    smoke = st.selectbox("Fuma:", ["no", "yes"])
    family = st.selectbox("Antecedentes Familiares:", ["no", "yes"])
    predict_btn = st.button("Calcular IMC / Predicción")
# --- DISEÑO DE COLUMNAS PRINCIPALES ---
col_main, col_side = st.columns([1, 1])
with col_side:
    # Módulo de Inferencia (Visualmente igual al mockup)
    st.markdown("""
        <div class="mockup-card">
            <div class="card-title">⚙️ Módulo de Inferencia</div>
            <ul style="list-style-type: none; padding-left: 0;">
                <li style="margin-bottom: 10px;">🔵 Procesando Datos...</li>
                <li style="margin-bottom: 10px;">🔵 Modelo Predictivo</li>
                <li style="margin-bottom: 10px;">🔵 Clasificación de Obesidad</li>
                <li style="margin-bottom: 10px;">🔵 API de Inferencia</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    if predict_btn:
        # Simulación de carga
        with st.spinner('Realizando inferencia...'):
            time.sleep(1.5)
            
        imc = peso / (altura ** 2)
        
        # Lógica de predicción simulada
        if imc < 18.5: prediction = "Peso Insuficiente"
        elif imc < 25: prediction = "Normal"
        elif imc < 30: prediction = "Sobrepeso Nivel I"
        elif imc < 35: prediction = "Obesidad Tipo I"
        else: prediction = "Obesidad Tipo II/III"
        # Resultado de Evaluación (Visualmente igual al mockup)
        st.markdown(f"""
            <div class="mockup-card">
                <div class="card-title" style="text-align: center; color: #1a3a5a;">Resultado de Evaluación</div>
                <h2 style="text-align: center; margin-bottom: 5px;">{prediction}</h2>
                <div class="risk-bar">
                    <div class="segment normal">Normal</div>
                    <div class="segment riesgo">Riesgo</div>
                    <div class="segment alerta">Alerta</div>
                    <div class="segment obesidad">Obesidad</div>
                </div>
                <p style="margin-top: 15px; font-weight: bold;">IMC: {imc:.1f}</p>
                <p style="font-size: 0.9rem; font-style: italic;">
                    <b>Recomendación:</b> Se recomienda mejorar su dieta y aumentar la actividad física.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Complete los datos y presione 'Calcular' para ver el diagnóstico.")
# --- DASHBOARD (PARTE INFERIOR - COMO EN LA IMAGEN) ---
st.markdown("---")
st.markdown('<div class="card-title">Tablero Analítico</div>', unsafe_allow_html=True)
dash_cols = st.columns(4)
with dash_cols[0]:
    # Distribución de Obesidad (Donut Chart)
    fig_donut = go.Figure(data=[go.Pie(
        labels=['Bajo Peso', 'Normal', 'Sobrepeso', 'Obesidad'],
        values=[15, 35, 25, 25],
        hole=.6,
        marker_colors=['#4682B4', '#9ACD32', '#FFD700', '#CD5C5C']
    )])
    fig_donut.update_layout(title_text="Distribución", height=300, showlegend=False)
    st.plotly_chart(fig_donut, use_container_width=True)
with dash_cols[1]:
    # FAF vs IMC (Scatter Plot)
    np.random.seed(42)
    df_scatter = pd.DataFrame({
        'FAF': np.random.uniform(0, 3, 50),
        'IMC': np.random.uniform(18, 40, 50),
        'Country': np.random.choice(['Mexico', 'Colombia', 'Peru'], 50)
    })
    fig_scatter = px.scatter(df_scatter, x='FAF', y='IMC', color='Country', height=300)
    fig_scatter.update_layout(title="FAF vs IMC")
    st.plotly_chart(fig_scatter, use_container_width=True)
with dash_cols[2]:
    # Importancia de Variables (Bar Chart)
    importancias = {'Peso': 90, 'Alim': 75, 'Activ': 60, 'Agua': 45, 'Edad': 30}
    df_imp = pd.DataFrame(importancias.items(), columns=['Var', 'Val'])
    fig_imp = px.bar(df_imp, x='Var', y='Val', color='Var', height=300)
    fig_imp.update_layout(title="Importancia", showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)
with dash_cols[3]:
    # Matriz de Confusión (Heatmap simplificado)
    z = [[10, 1], [2, 17]]
    fig_heat = px.imshow(z, text_auto=True, color_continuous_scale='Blues',
                         labels=dict(x="Predicho", y="Real"),
                         x=['Error', 'Precisión'], y=['Error', 'Precisión'], height=300)
    fig_heat.update_layout(title="Desempeño")
    st.plotly_chart(fig_heat, use_container_width=True)
