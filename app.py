"""Streamlit app para clasificar imágenes en Gato, Perro u Otro (humano/objeto)."""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# Reducir verbosidad de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(
    page_title="Detector de Mascotas",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_PATH = "data/modelo_multiclase.keras"
IMAGE_SIZE: Tuple[int, int] = (150, 150)
CLASS_NAMES: Dict[int, str] = {
    0: "Gato 🐱",
    1: "Perro 🐶",
    2: "No es Mascota (Otro) 👤",
}


@st.cache_resource(show_spinner=False)
def load_predictive_model(model_path: str = MODEL_PATH):
    """Carga y cachea el modelo de clasificación multiclase."""
    return load_model(model_path)


def prepare_image(image: Image.Image) -> np.ndarray:
    """Normaliza y redimensiona la imagen para el modelo."""
    processed = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(processed, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(image: Image.Image) -> Tuple[int, float, np.ndarray]:
    """Devuelve la clase predicha, la confianza y el vector completo de probabilidades."""
    model = load_predictive_model()
    batch = prepare_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    class_idx = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    return class_idx, confidence, probabilities


def render_sidebar() -> None:
    """Instrucciones y tips permanentes en la barra lateral."""
    st.sidebar.header("Cómo usar la app")
    st.sidebar.markdown(
        """
        1. Usa la cámara o sube una foto clara.
        2. Procura que la mascota ocupe buena parte de la imagen.
        3. Si aparece un humano u objeto, la app lo marcará como "No es Mascota".
        4. Revisa la sección de debug para entender la decisión del modelo.
        """
    )
    st.sidebar.info("El modelo se recalibra automáticamente cuando subes nuevas fotos.")


def render_results(image: Image.Image, class_idx: int, confidence: float, probabilities: np.ndarray) -> None:
    """Muestra resultados, imagen y panel de depuración."""
    confidence_pct = confidence * 100

    col_image, col_result = st.columns([1.1, 1])
    with col_image:
        st.image(image, caption="Imagen evaluada", width="stretch")

    with col_result:
        if class_idx == 2:
            st.warning("⚠️ Objeto/Humano detectado (No es mascota)")
        else:
            label = CLASS_NAMES[class_idx]
            st.success(f"DETECTADO: {label.upper()}")

        st.metric(label="Confianza", value=f"{confidence_pct:.1f}%")
        st.progress(int(confidence_pct))

    with st.expander("Ver detalles técnicos (Debug)"):
        prob_percent = probabilities * 100
        df_probs = pd.DataFrame(
            {
                "Probabilidad (%)": prob_percent,
            },
            index=[CLASS_NAMES[idx] for idx in CLASS_NAMES],
        )
        st.bar_chart(df_probs)
        st.write("Vector de probabilidades:", df_probs.T)


def process_image_input(image_input) -> None:
    """Orquesta el flujo: apertura de imagen, predicción y render."""
    if image_input is None:
        return
    try:
        image = Image.open(image_input)
    except Exception as exc:  # pragma: no cover
        st.error(f"No se pudo procesar la imagen: {exc}")
        return

    class_idx, confidence, probabilities = predict(image)
    render_results(image, class_idx, confidence, probabilities)


def render_header() -> None:
    """Encabezado principal."""
    st.title("🐾 Detector Inteligente de Mascotas")
    st.caption("Clasifica entre Gato, Perro o indica si no hay una mascota.")


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    render_sidebar()
    render_header()

    tab_camera, tab_upload = st.tabs(["📷 Usar cámara", "📤 Subir foto"])

    with tab_camera:
        st.subheader("Captura con tu cámara")
        captured = st.camera_input("Cámara", label_visibility="collapsed")
        if captured is not None:
            with st.spinner("Analizando imagen..."):
                process_image_input(captured)

    with tab_upload:
        st.subheader("Sube un archivo de imagen")
        uploaded = st.file_uploader(
            "Selecciona un archivo",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            with st.spinner("Procesando imagen..."):
                process_image_input(uploaded)


if __name__ == "__main__":
    main()
