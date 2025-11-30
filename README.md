<div align="center">
  <img src="https://img.icons8.com/color/96/000000/dog.png" alt="Dogs" width="80"/>
  <img src="https://img.icons8.com/color/96/000000/cat.png" alt="Cats" width="80"/>
  <h1>Detector de Mascotas con IA</h1>
  <h3>Clasificador de imágenes con Redes Neuronales Convolucionales</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://detector-mascotas.streamlit.app/)
</div>

## 📊 Resumen del Proyecto

Sistema de visión por computadora que clasifica imágenes en tres categorías: perros, gatos u otros objetos, utilizando técnicas avanzadas de Deep Learning. El modelo logra una precisión superior al 97% gracias a la implementación de transfer learning con MobileNetV2.

<div align="center">
  <img src="https://img.shields.io/badge/Accuracy-97.30%25-brightgreen" alt="Accuracy">
  <img src="https://img.shields.io/badge/Precision-97.58%25-brightgreen" alt="Precision">
  <img src="https://img.shields.io/badge/Recall-97.77%25-brightgreen" alt="Recall">
  <img src="https://img.shields.io/badge/F1_Score-97.34%25-brightgreen" alt="F1 Score">
</div>

## 🚀 Características Principales

- 🎯 **Alta Precisión**: Más del 97% de precisión en la clasificación
- 🖼️ **Tres Categorías**: Clasifica entre perros, gatos y otras imágenes
- 📱 **Interfaz Web**: Fácil de usar con Streamlit
- 🔍 **Panel de Análisis**: Visualización detallada de predicciones
- ⚡ **Rendimiento Optimizado**: Inferencia rápida con MobileNetV2

## 📈 📊 Métricas del Modelo

### Rendimiento General
- **Exactitud (Accuracy)**: 97.30%
- **Precisión Promedio**: 97.58%
- **Sensibilidad (Recall) Promedio**: 97.77%
- **Puntuación F1 Promedio**: 97.34%

### Desempeño por Clase
| Categoría  | Precisión | Sensibilidad | F1-Score | Soporte |
|------------|-----------|--------------|----------|---------|
| 🐱 Gatos   | 100.00%   | 97.14%       | 98.55%   | 70      |
| 🐶 Perros  | 89.74%    | 100.00%      | 94.59%   | 70      |
| 🧸 Otros   | 100.00%   | 96.15%       | 98.04%   | 156     |

### Análisis de las Métricas
1. **Precisión** (Valores Predictivos Positivos):
   - Gatos y Otros: 100% - Excelente capacidad para identificar correctamente las clases positivas
   - Perros: 89.74% - Algunos falsos positivos (clasifica como perros algunas imágenes que no lo son)

2. **Sensibilidad** (Tasa de Verdaderos Positivos):
   - Perros: 100% - Detecta correctamente todos los perros
   - Gatos: 97.14% - Muy buena detección
   - Otros: 96.15% - Excelente capacidad de generalización

3. **F1-Score** (Media Armónica):
   - Valores superiores al 94% en todas las clases
   - Balance óptimo entre precisión y sensibilidad

4. **Soporte**:
   - Clase "Otros" tiene más del doble de muestras que las demás
   - El modelo maneja bien el desbalance de clases

### Matriz de Confusión
Predicción Gato Perro Otro

Real Gato 68 2 0 Perro 0 70 0 Otro 3 3 150


### Interpretación:
- **Fortalezas**:
  - Excelente rendimiento general (97.3% de exactitud)
  - Perfecta precisión en las clases Gato y Otros
  - Detección perfecta de perros (100% de sensibilidad)

- **Áreas de Mejora**:
  - Algunos falsos positivos en la clase Perro
  - Pequeña confusión entre Gato y Otra

### Desempeño por Clase
| Categoría  | Precisión | Sensibilidad | F1-Score | Soporte |
|------------|-----------|--------------|----------|---------|
| 🐱 Gatos   | 100.00%   | 97.14%       | 98.55%   | 70      |
| 🐶 Perros  | 89.74%    | 100.00%      | 94.59%   | 70      |
| 🧸 Otros   | 100.00%   | 96.15%       | 98.04%   | 156     |

### Arquitectura del Modelo
- **Backbone**: MobileNetV2 con pesos pre-entrenados en ImageNet
- **Capas Adicionales**:
  - GlobalAveragePooling2D
  - Densa (128 neuronas, ReLU)
  - Dropout (0.5)
  - Capa de salida con activación Softmax

## 🛠️ Instalación


# 1. Clonar el repositorio
git clone [https://github.com/veraguillen/Detector-de-Mascotas-con-IA--CNN-.git](https://github.com/veraguillen/Detector-de-Mascotas-con-IA--CNN-.git)
cd Detector-de-Mascotas-con-IA--CNN-

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt


## 🚀 Uso

-Entrenamiento del Modelo
-python train_cnn.py
-Ejecutar la Aplicación Web
-streamlit run app.py
-Acceso en Línea

# La aplicación está disponible en:
🔗 https://detector-mascotas.streamlit.app/

# 🏗️ Estructura del Proyecto
.
├── data/                    # Conjunto de datos
│   ├── train/               # Imágenes de entrenamiento
│   └── test/                # Imágenes de prueba
├── models/                  # Modelos guardados
├── results/                 # Resultados y métricas
├── app.py                   # Aplicación Streamlit
├── train_cnn.py             # Entrenamiento del modelo
├── analyze_metrics.py       # Análisis de métricas
└── requirements.txt         # Dependencias


## 👨‍💻 Sobre el Autor

**Vera Guillén**

-   **Portfolio:** **[vera-guillen.vercel.app](https://vera-guillen.vercel.app/)**
-   **GitHub:** [@veraguillen](https://github.com/veraguillen)
-   **LinkedIn:** [https://www.linkedin.com/in/vera-guillen-9b464a303/]