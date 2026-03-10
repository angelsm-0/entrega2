import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="Malnutricion",
    page_icon="🏥",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

# ------------------ CSS ------------------

st.markdown("""
<style>

.main {
    background-color: #f0f4f8;
}

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

</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------

st.sidebar.markdown("## Datos del Usuario")

edad = st.sidebar.number_input("Edad", 1, 100, 25)

genero = st.sidebar.selectbox(
    "Género",
    ["Female", "Male"]
)

altura = st.sidebar.number_input("Altura (m)", 1.3, 2.3, 1.70)
peso = st.sidebar.number_input("Peso (kg)", 30, 200, 70)

st.sidebar.markdown("### Hábitos")

family = st.sidebar.selectbox(
    "Historial familiar obesidad",
    ["yes", "no"]
)

favc = st.sidebar.selectbox(
    "Consumo frecuente comida calórica",
    ["yes", "no"]
)

fcvc = st.sidebar.slider("Consumo vegetales", 1, 3, 2)

ncp = st.sidebar.slider("Número comidas", 1, 4, 3)

caec = st.sidebar.selectbox(
    "Comida entre comidas",
    ["no", "Sometimes", "Frequently", "Always"]
)

smoke = st.sidebar.selectbox(
    "Fuma",
    ["yes", "no"]
)

ch2o = st.sidebar.slider("Consumo agua", 1, 3, 2)

scc = st.sidebar.selectbox(
    "Monitorea calorías",
    ["yes", "no"]
)

faf = st.sidebar.slider("Actividad física", 0, 3, 1)

tue = st.sidebar.slider("Uso tecnología", 0, 3, 1)

calc = st.sidebar.selectbox(
    "Consumo alcohol",
    ["no", "Sometimes", "Frequently", "Always"]
)

mtrans = st.sidebar.selectbox(
    "Transporte",
    [
        "Public_Transportation",
        "Walking",
        "Automobile",
        "Bike",
        "Motorbike"
    ]
)

predict_btn = st.sidebar.button("Realizar Predicción")

# ------------------ LAYOUT ------------------

col_main, col_side = st.columns([1,1])

# ------------------ INFERENCIA ------------------

with col_side:

    st.markdown("""
    <div class="mockup-card">
        <div class="card-title">⚙️ Módulo de Inferencia</div>
        <ul>
        <li>Procesando datos</li>
        <li>Pipeline ML</li>
        <li>Clasificación</li>
        <li>API FastAPI</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:

        with st.spinner("Consultando modelo..."):
            time.sleep(1)

        data = {
            "Gender": genero,
            "Age": edad,
            "Height": altura,
            "Weight": peso,
            "family_history_with_overweight": family,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }

        try:

            r = requests.post(API_URL, json=data)

            resultado = r.json()

            pred = resultado["prediction_label"]

            imc = peso / (altura**2)

            st.markdown(f"""
            <div class="mockup-card" style="border-left: 5px solid #2e7d32;">
            <h2 style="text-align:center;">Resultado</h2>

            <h3 style="text-align:center;">{pred}</h3>

            <div class="risk-bar">
            <div class="segment normal">Normal</div>
            <div class="segment riesgo">Riesgo</div>
            <div class="segment alerta">Alerta</div>
            <div class="segment obesidad">Obesidad</div>
            </div>

            <p style="margin-top:15px"><b>IMC:</b> {imc:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        except:

            st.error("Error conectando con la API")

    else:

        st.info("Ingrese datos y presione el botón")

# ------------------ DASHBOARD ------------------

st.markdown("---")
st.markdown("## Tablero Analítico")

dash_cols = st.columns(4)

with dash_cols[0]:

    fig = go.Figure(data=[go.Pie(
        labels=['Bajo Peso','Normal','Sobrepeso','Obesidad'],
        values=[15,35,25,25],
        hole=.6
    )])

    fig.update_layout(title="Distribución")

    st.plotly_chart(fig, use_container_width=True)

with dash_cols[1]:

    np.random.seed(42)

    df = pd.DataFrame({
        "FAF": np.random.uniform(0,3,50),
        "IMC": np.random.uniform(18,40,50)
    })

    fig = px.scatter(df, x="FAF", y="IMC")

    st.plotly_chart(fig, use_container_width=True)

with dash_cols[2]:

    imp = pd.DataFrame({
        "Variable":["Weight","FAF","FCVC","Age"],
        "Valor":[90,70,60,40]
    })

    fig = px.bar(imp, x="Variable", y="Valor")

    st.plotly_chart(fig, use_container_width=True)

with dash_cols[3]:

    z = [[10,1],[2,17]]

    fig = px.imshow(
        z,
        text_auto=True,
        labels=dict(x="Predicho", y="Real")
    )

    st.plotly_chart(fig, use_container_width=True)