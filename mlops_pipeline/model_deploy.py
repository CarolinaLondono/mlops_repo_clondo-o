
"""API para realizar predicciones de comportamiento de pago."""

from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from mlops_pipeline.ft_engineering import (
    COLS_NUMERICAS,
    COLS_SALDOS,
    COLS_CATEGORICAS,
    COLS_BINARIAS,
    COL_ORDINAL,
    limpiar_datos,
)

import sys
from mlops_pipeline import ft_engineering

# Compatibilidad con el nombre utilizado al guardar el modelo
sys.modules.setdefault("ft_engineering", ft_engineering)


app = FastAPI(
    title="API de predicción de créditos",
    description="Predicción de pago a tiempo o impago.",
    version="1.0.0",
)

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_final.joblib"

# Variables utilizadas por el modelo final
COLUMNAS = (
    COLS_NUMERICAS
    + COLS_SALDOS
    + COLS_CATEGORICAS
    + COLS_BINARIAS
    + [COL_ORDINAL]
)

modelo = None


def obtener_modelo():
    global modelo
    if modelo is None:
        if not RUTA_MODELO.exists():
            raise HTTPException(
                status_code=503,
                detail="No se encontró modelo_final.joblib. Entrena el modelo primero.",
            )
        modelo = joblib.load(RUTA_MODELO)
    return modelo


def preparar_datos(df):
    if df.empty:
        raise HTTPException(400, "No se recibieron registros.")

    faltantes = [c for c in COLUMNAS if c not in df.columns]
    if faltantes:
        raise HTTPException(
            422,
            detail={"columnas_faltantes": faltantes},
        )

    df = df[COLUMNAS].copy()

    # La función original elimina estas dos columnas.
    df["fecha_prestamo"] = np.nan
    df["saldo_mora_codeudor"] = np.nan

    try:
        for col in COLS_NUMERICAS + COLS_SALDOS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["tipo_credito"] = pd.to_numeric(
            df["tipo_credito"], errors="coerce"
        )

        df = limpiar_datos(df)
        return df

    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Error preparando datos: {error}",
        ) from error


@app.get("/")
def inicio():
    return {
        "mensaje": "API de créditos funcionando",
        "documentacion": "/docs",
    }


@app.post("/predict")
async def predecir(request: Request):
    tipo = request.headers.get("content-type", "")

    try:
        if "application/json" in tipo:
            contenido = await request.json()

            if isinstance(contenido, dict):
                registros = contenido.get("records", [contenido])
            elif isinstance(contenido, list):
                registros = contenido
            else:
                raise ValueError("Formato JSON no válido.")

            if not isinstance(registros, list):
                raise ValueError("'records' debe ser una lista.")

            df = pd.DataFrame(registros)

        elif "multipart/form-data" in tipo:
            formulario = await request.form()
            archivo = formulario.get("file")

            if archivo is None:
                raise ValueError("Se requiere un archivo llamado 'file'.")

            contenido = await archivo.read()
            df = pd.read_csv(StringIO(contenido.decode("utf-8-sig")))

        elif "text/csv" in tipo:
            contenido = await request.body()
            df = pd.read_csv(StringIO(contenido.decode("utf-8-sig")))

        else:
            raise HTTPException(
                status_code=415,
                detail="Utiliza JSON o CSV.",
            )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Error leyendo la entrada: {error}",
        ) from error

    datos = preparar_datos(df)
    modelo_cargado = obtener_modelo()

    try:
        predicciones = modelo_cargado.predict(datos)
        probabilidades = modelo_cargado.predict_proba(datos)

        clases = list(modelo_cargado.classes_)
        indice_impago = clases.index(0)

        resultados = []

        for pred, proba in zip(predicciones, probabilidades):
            resultados.append({
                "prediccion": int(pred),
                "resultado": (
                    "Pago a tiempo" if int(pred) == 1 else "Impago"
                ),
                "probabilidad_impago": round(
                    float(proba[indice_impago]), 4
                ),
            })

        return {
            "total_registros": len(resultados),
            "predicciones": resultados,
        }

    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"No fue posible realizar la predicción: {error}",
        ) from error
