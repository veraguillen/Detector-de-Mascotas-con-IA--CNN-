# 🐶 Detector de Mascotas con IA (CNN)

¡Bienvenido a mi aplicación de clasificación de imágenes! Este proyecto utiliza **Deep Learning** para identificar si una imagen contiene un **Perro**, un **Gato**, o si **No es una mascota**.

La aplicación ha sido entrenada con una Red Neuronal Convolucional (CNN) personalizada y desplegada utilizando **Streamlit** para una experiencia web responsiva (móvil y escritorio).

## ✨ Características Principales

- 🔍 Clasificación en 3 categorías: Perro, Gato u Otro (humano/objeto)
- 📱 Interfaz responsiva que funciona en móviles y escritorio
- 📊 Muestra métricas de confianza y visualización de probabilidades
- 🎯 Precisión mejorada con umbral de confianza ajustable
- 🛠️ Panel de depuración integrado para análisis detallado

## 🎮 Uso

1. Abre la aplicación en tu navegador (se abrirá automáticamente al ejecutar `streamlit run app.py`)
2. Selecciona una de las opciones:
   - 📸 Usar la cámara para tomar una foto
   - 📁 Subir una imagen desde tu dispositivo
3. La IA analizará la imagen y mostrará el resultado con un porcentaje de confianza
4. Usa la pestaña de "Ver detalles técnicos" para entender mejor la decisión del modelo

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

El modelo es una Red Neuronal Convolucional (CNN) entrenada desde cero con las siguientes características:

- **Arquitectura:**
  - Capas convolucionales con activación ReLU
  - Capas MaxPooling2D para reducción dimensional
  - Capa de aplanamiento
  - Capas densas con Dropout para regularización
  - Función de activación Softmax en la capa de salida

- **Preprocesamiento:**
  - Redimensionamiento a 150x150 píxeles
  - Normalización de valores de píxeles a [0, 1]
  - Aumento de datos durante el entrenamiento

## 🌐 Despliegue

La aplicación está diseñada para desplegarse fácilmente en Streamlit Cloud.
Desarrollado con ❤️ por Vera Guillen


