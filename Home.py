
import streamlit as st

from modelo.analisis_depresivo import Analizador


# Configuración
st.set_page_config(
    page_title="Analizador de Sentimientos",
    page_icon="🧠",
    layout="centered"
)

# Inicializar modelo una sola vez
@st.cache_resource
def cargar_modelo():
    return Analizador()


analizar = cargar_modelo()


# Interfaz
st.title("🧠 Analizador de Sentimientos")

st.write(
    "Ingresa un comentario para analizar su sentimiento."
)

comentario = st.text_area(
    "Escribe tu comentario:",
    placeholder="¿Cómo te has sentido hoy?",
    height=150
)

if st.button("Analizar sentimiento", type="primary"):

    if comentario.strip():

        with st.spinner("Analizando comentario..."):

            try:
                resultado = analizar.analizar(comentario)

                st.success("Análisis completado")

                st.subheader("Resultado")
                st.write(resultado)

            except Exception as e:
                st.error(f"Error durante el análisis: {e}")

    else:
        st.warning("Por favor, ingresa un comentario.")
