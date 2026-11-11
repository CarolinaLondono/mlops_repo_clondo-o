
"""Monitoreo de cambios en las distribuciones de los datos."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import ks_2samp

from ft_engineering import cargar_y_dividir, limpiar_datos

st.set_page_config(
    page_title="Monitoreo de créditos",
    page_icon="📊",
    layout="wide"
)

st.title("Monitoreo de datos - Modelo de créditos")
st.write(
    "Comparación de los datos históricos y actuales "
    "para identificar posibles cambios en sus distribuciones."
)

# Se reutiliza la división definida durante el entrenamiento.
@st.cache_data
def obtener_datos():
    X_train, X_test, _, _ = cargar_y_dividir()
    return X_train, X_test


historicos, evaluacion = obtener_datos()

archivo = st.file_uploader(
    "Opcional: cargar nuevos datos en formato CSV",
    type=["csv"]
)

if archivo is not None:
    try:
        actuales = limpiar_datos(pd.read_csv(archivo))
        st.info("Analizando los datos del CSV cargado.")
    except Exception as error:
        st.error(f"No fue posible procesar el archivo: {error}")
        st.stop()
else:
    actuales = evaluacion
    st.info(
        "Modo demostración: se utilizan los datos de evaluación. "
        "No representan datos nuevos de producción."
    )

variables = [
    "capital_prestado",
    "cuota_pactada",
    "salario_cliente",
    "edad_cliente",
    "plazo_meses",
    "puntaje_datacredito",
]

st.subheader("Análisis de Data Drift")

resultados = []
datos_validos = {}

for variable in variables:
    if variable not in actuales.columns:
        continue

    referencia = pd.to_numeric(
        historicos[variable], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    nueva = pd.to_numeric(
        actuales[variable], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if referencia.empty or nueva.empty:
        continue

    estadistico, p_valor = ks_2samp(referencia, nueva)

    alerta = p_valor < 0.05

    resultados.append({
        "Variable": variable,
        "Estadístico KS": round(estadistico, 4),
        "P-valor": round(p_valor, 6),
        "Estado": "Alerta" if alerta else "Sin alerta"
    })

    datos_validos[variable] = (referencia, nueva)

if not resultados:
    st.error("No hay variables suficientes para realizar el análisis.")
    st.stop()

tabla = pd.DataFrame(resultados)

total_alertas = int((tabla["Estado"] == "Alerta").sum())

col1, col2, col3 = st.columns(3)
col1.metric("Variables evaluadas", len(tabla))
col2.metric("Variables con alerta", total_alertas)
col3.metric("Nivel de significancia", "5 %")

st.subheader("Resultados de las pruebas estadísticas")
st.dataframe(tabla, use_container_width=True, hide_index=True)

if total_alertas > 0:
    st.warning(
        f"Se detectaron cambios estadísticamente significativos "
        f"en {total_alertas} variable(s)."
    )
else:
    st.success(
        "No se detectaron cambios estadísticamente significativos "
        "en las variables evaluadas."
    )

st.subheader("Comparación de distribuciones")

variable_elegida = st.selectbox(
    "Selecciona una variable",
    list(datos_validos.keys())
)

referencia, nueva = datos_validos[variable_elegida]

fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(
    referencia, bins=35, density=True,
    alpha=0.5, label="Históricos"
)
ax.hist(
    nueva, bins=35, density=True,
    alpha=0.5, label="Actuales"
)

ax.set_title(f"Distribución de {variable_elegida}")
ax.set_xlabel(variable_elegida)
ax.set_ylabel("Densidad")
ax.legend()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "El test Kolmogorov-Smirnov compara distribuciones numéricas. "
    "Un p-valor menor a 0.05 activa una alerta estadística, "
    "pero no demuestra por sí solo una pérdida de rendimiento del modelo."
)
