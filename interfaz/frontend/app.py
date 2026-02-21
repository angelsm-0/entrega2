import streamlit as st
import requests
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Obesity Predictor", layout="wide")
st.title("Dashboard Predictivo - Nivel de Obesidad")

st.sidebar.header("Ingreso de Datos")

edad = st.sidebar.slider("Edad", 15, 65, 25)
altura = st.sidebar.slider("Altura (m)", 1.4, 2.0, 1.70)
peso = st.sidebar.slider("Peso (kg)", 40, 150, 70)
faf = st.sidebar.slider("Actividad Física (0-4)", 0, 4, 2)
ch2o = st.sidebar.slider("Consumo Agua (1-3)", 1, 3, 2)
fcvc = st.sidebar.slider("Consumo Vegetales (1-3)", 1, 3, 2)

if st.sidebar.button("Predecir"):

   payload = {
       "edad": edad,
       "altura": altura,
       "peso": peso,
       "faf": faf,
       "ch2o": ch2o,
       "fcvc": fcvc
   }

   response = requests.post(API_URL, json=payload)

   if response.status_code == 200:
       result = response.json()
       prediction = result["prediction"]

       imc = peso / (altura ** 2)

       st.subheader("Resultado")

       col1, col2 = st.columns(2)

       with col1:
           st.metric("Nivel Predicho", prediction)
           st.metric("IMC", round(imc, 2))

       with col2:
           fig = go.Figure(go.Indicator(
               mode="gauge+number",
               value=imc,
               title={'text': "IMC"},
               gauge={'axis': {'range': [10, 50]}}
           ))
           st.plotly_chart(fig)

   else:
       st.error("Error al conectar con la API")
