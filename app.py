"""Streamlit app para clasificar imágenes en Gato, Perro u Otro (humano/objeto)."""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# Reducir verbosidad de TensorFlow para mantener la consola limpia
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(
    page_title="Detector de Mascotas",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CONFIGURACIÓN CRÍTICA ACTUALIZADA ---
# Apuntamos al nuevo modelo de alta precisión
MODEL_PATH = "data/modelo_perros_pro.keras"
# MobileNetV2 requiere estrictamente 224x224 píxeles
IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Mapeo de clases basado en el entrenamiento (orden alfabético: cats, dogs, others)
CLASS_NAMES: Dict[int, str] = {
    0: "Gato 🐱",
    1: "Perro 🐶",
    2: "No es Mascota (Otro) 👤",
}


@st.cache_resource(show_spinner=False)
def load_predictive_model(model_path: str = MODEL_PATH):
    """Carga y cachea el modelo de clasificación multiclase."""
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error crítico cargando el modelo: {e}")
        st.stop()


def prepare_image(image: Image.Image) -> np.ndarray:
    """Normaliza y redimensiona la imagen para el modelo MobileNetV2."""
    # 1. Convertir a RGB (evita errores con PNGs transparentes)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # 2. Redimensionar a 224x224 (CRÍTICO)
    processed = image.resize(IMAGE_SIZE)
    
    # 3. Convertir a array y Normalizar (0 a 1)
    array = np.asarray(processed, dtype=np.float32) / 255.0
    
    # 4. Expandir dimensiones (de (224,224,3) a (1,224,224,3)) para Keras
    return np.expand_dims(array, axis=0)


def predict(image: Image.Image) -> Tuple[int, float, np.ndarray]:
    """Devuelve la clase predicha, la confianza y el vector completo de probabilidades."""
    model = load_predictive_model()
    batch = prepare_image(image)
    
    # Predecir
    probabilities = model.predict(batch, verbose=0)[0]
    
    class_idx = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    return class_idx, confidence, probabilities


def render_sidebar() -> None:
    """Instrucciones y tips permanentes en la barra lateral."""
    st.sidebar.header("Cómo usar la app")
    st.sidebar.markdown(
        """
        1. **Cámara:** Ideal para móviles.
        2. **Subir:** Para fotos guardadas.
        3. **Luz:** Procura que haya buena iluminación.
        
        ⚠️ **Nota:** El modelo ha sido entrenado para distinguir humanos de mascotas.
        """
    )
    st.sidebar.info("Modelo: MobileNetV2 (Transfer Learning)")


def render_results(image: Image.Image, class_idx: int, confidence: float, probabilities: np.ndarray) -> None:
    """Muestra resultados, imagen y panel de depuración."""
    confidence_pct = confidence * 100

    # Layout responsivo: columnas en PC, apilado en móvil
    col_image, col_result = st.columns([1, 1])
    
    with col_image:
        # CORRECCIÓN: width="stretch" está deprecado, usamos use_container_width
        st.image(image, caption="Imagen capturada", use_container_width=True)

    with col_result:
        # Lógica de visualización
        if class_idx == 2: # Clase 'Others'
            st.warning("⚠️ **ALERTA:** No se detecta mascota.")
            st.markdown("### Parece ser: **Humano u Objeto**")
        else:
            label = CLASS_NAMES[class_idx]
            st.success(f"✅ **DETECTADO:** {label}")
            
        # Métricas grandes
        st.metric(label="Confianza de la IA", value=f"{confidence_pct:.1f}%")
        st.progress(int(confidence_pct))
        
        # Mensaje de duda si la confianza es baja (aunque el modelo pro suele ser muy seguro)
        if confidence < 0.60:
            st.info("🤔 La IA tiene dudas. Intenta acercarte más o mejorar la luz.")

    # Sección de Debug
    with st.expander("Ver detalles técnicos (Probabilidades)"):
        df_probs = pd.DataFrame(
            {
                "Confianza": probabilities,
            },
            index=[CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
        )
        st.bar_chart(df_probs)


def process_image_input(image_input) -> None:
    """Orquesta el flujo: apertura de imagen, predicción y render."""
    if image_input is None:
        return
    try:
        image = Image.open(image_input)
    except Exception as exc: 
        st.error(f"No se pudo leer el archivo: {exc}")
        return

    class_idx, confidence, probabilities = predict(image)
    render_results(image, class_idx, confidence, probabilities)


def render_header() -> None:
    """Encabezado principal."""
    st.title("🐾 Detector Inteligente de Mascotas")
    st.markdown("### Clasificación profesional con IA")


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    render_sidebar()
    render_header()

    tab_camera, tab_upload = st.tabs(["📷 Usar cámara", "📤 Subir foto"])

    with tab_camera:
        # Eliminamos el label visible para limpiar la UI móvil
        captured = st.camera_input("Tomar foto", label_visibility="collapsed")
        if captured is not None:
            with st.spinner("Analizando con MobileNet..."):
                process_image_input(captured)

    with tab_upload:
        st.write("Sube una imagen de tu galería:")
        uploaded = st.file_uploader(
            "Selecciona un archivo",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            with st.spinner("Procesando imagen..."):
                process_image_input(uploaded)


if __name__ == "__main__":
    main()