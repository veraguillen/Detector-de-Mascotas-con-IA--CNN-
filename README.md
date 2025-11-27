# 🐶 Detector de Mascotas con IA (CNN)

¡Bienvenido a mi aplicación de clasificación de imágenes! Este proyecto utiliza **Deep Learning** para identificar si una imagen contiene un **Perro**, un **Gato**, o si **No es una mascota**.

La aplicación ha sido entrenada con una Red Neuronal Convolucional (CNN) personalizada y desplegada utilizando **Streamlit** para una experiencia web responsiva (móvil y escritorio).

## ✨ Características Principales

- 🔍 Clasificación en 3 categorías: Perro, Gato u Otro (humano/objeto)
- 📱 Interfaz responsiva que funciona en móviles y escritorio
- 📊 Muestra métricas de confianza y visualización de probabilidades
- 🎯 Precisión mejorada con umbral de confianza ajustable
- 🛠️ Panel de depuración integrado para análisis detallado

## 🛠️ Tecnologías Usadas

- **Python 3.10+**
- **TensorFlow / Keras:** Para la construcción y entrenamiento del modelo CNN.
- **Streamlit:** Para la interfaz web (Frontend).
- **Pillow / NumPy:** Para el procesamiento de imágenes.

## 📦 Instalación y Uso Local

Si quieres correr este proyecto en tu computadora:

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/veraguillen/Detector-de-Mascotas-con-IA--CNN-.git
   cd detector-mascotas
Crea un entorno virtual e instala dependencias:
code
Bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
Ejecuta la aplicación:
code
Bash
streamlit run app.py


## 🧠 Sobre el Modelo

El modelo es una CNN entrenada desde cero con las siguientes características:
Entrada: Imágenes redimensionadas a 150x150 píxeles.
Normalización: Valores de píxeles escalados a [0, 1].
Capa de salida: Softmax con 3 neuronas (Gato, Perro, Otro).

## 🌐 Despliegue

La aplicación está diseñada para desplegarse fácilmente en Streamlit Cloud.
Desarrollado con ❤️ por Vera Guillen


