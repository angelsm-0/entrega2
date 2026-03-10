# Manual de Instalación — Tablero de Predicción de Obesidad

> Guía paso a paso para instalar y ejecutar el tablero en tu entorno local.

---

## Requisitos del Sistema

| Requisito | Detalle |
|-----------|---------|
| Python | 3.9 o superior |
| Git | Cualquier versión reciente |
| Conexión a internet | Requerida para clonar el repositorio e instalar dependencias |
| Sistema operativo | Windows, Linux o macOS |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/angelsm-0/entrega2.git
cd entrega2
```

### 2. Crear un entorno virtual (recomendado)

```bash
python -m venv venv
```

Activar el entorno según tu sistema operativo:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Esto instalará todas las librerías necesarias para ejecutar el proyecto.

---

## Ejecutar el Tablero

```bash
streamlit run interfaz/app.py
```

El sistema abrirá automáticamente el tablero en tu navegador en:

```
http://localhost:8501
```

---

## Estructura del Proyecto

```
entrega2/
│
├── interfaz/        → Aplicación Streamlit
├── data/            → Datos utilizados por el modelo
├── models/          → Modelos entrenados
├── notebooks/       → Análisis exploratorio y entrenamiento
└── requirements.txt → Dependencias del proyecto
```

---

## Solución de Problemas

**Error: `'streamlit' no se reconoce como un comando interno o externo`**

```bash
pip install streamlit
```

Luego vuelve a ejecutar la aplicación:

```bash
streamlit run interfaz/app.py
```

---

## Licencia

Este proyecto fue desarrollado con fines académicos.
