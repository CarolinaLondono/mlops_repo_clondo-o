"""ft_engineering.py
Limpieza de datos, pipeline de transformaciones y división train/test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

RUTA_DATOS = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"
TARGET = "Pago_atiempo"

COLS_NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]
COLS_SALDOS = ["saldo_mora", "saldo_total", "saldo_principal"]
COLS_CATEGORICAS = ["tipo_credito"]
COLS_BINARIAS = ["tipo_laboral"]
COL_ORDINAL = "tendencia_ingresos"
ORDEN_TENDENCIA = ["sin_dato", "Decreciente", "Estable", "Creciente"]


class LimitarExtremos(BaseEstimator, TransformerMixin):
    """Limita cada columna al percentil indicado, calculado solo con los datos de
    entrenamiento. Si el percentil coincide con el mínimo (columna casi toda en 0),
    no se limita, para no perder los pocos casos con valor."""

    def __init__(self, percentil=0.99):
        self.percentil = percentil

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        limite = np.percentile(X, self.percentil * 100, axis=0)
        self.limites_ = np.where(limite > X.min(axis=0), limite, X.max(axis=0))
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.limites_)


def limpiar_datos(df):
    """Aplica las reglas de limpieza definidas en el EDA."""
    df = df.copy()
    validos = ["Creciente", "Decreciente", "Estable"]
    df.loc[~df[COL_ORDINAL].isin(validos), COL_ORDINAL] = np.nan
    df.loc[df["edad_cliente"] > 100, "edad_cliente"] = np.nan
    df.loc[df["puntaje_datacredito"].isin([0, -7, 7, 999]), "puntaje_datacredito"] = np.nan
    df.loc[df["salario_cliente"] == 0, "salario_cliente"] = np.nan
    df.loc[df["promedio_ingresos_datacredito"] == 0, "promedio_ingresos_datacredito"] = np.nan
    df["tipo_credito"] = np.where(
        df["tipo_credito"].isin([4, 9]), df["tipo_credito"].astype(str), "otros"
    )
    return df.drop(columns=["saldo_mora_codeudor", "fecha_prestamo"])


def construir_preprocesador(incluir_puntaje=True):
    """ColumnTransformer con las ramas numérica, saldos, categórica, binaria y ordinal."""
    cols_num = COLS_NUMERICAS + (["puntaje"] if incluir_puntaje else [])

    numericas = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("limite", LimitarExtremos(0.99)),
        ("scaler", MinMaxScaler()),
    ])
    saldos = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("limite", LimitarExtremos(0.99)),
        ("scaler", MinMaxScaler()),
    ])
    categoricas = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binarias = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[["Empleado", "Independiente"]])),
    ])
    ordinales = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="sin_dato")),
        ("ordinal", OrdinalEncoder(
            categories=[ORDEN_TENDENCIA],
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
    ])

    return ColumnTransformer([
        ("numericas", numericas, cols_num),
        ("saldos", saldos, COLS_SALDOS),
        ("categoricas", categoricas, COLS_CATEGORICAS),
        ("binarias", binarias, COLS_BINARIAS),
        ("ordinales", ordinales, [COL_ORDINAL]),
    ], remainder="drop")


def cargar_y_dividir(test_size=0.2, random_state=42):
    """Carga, limpia y divide en entrenamiento y evaluación (estratificado)."""
    df = limpiar_datos(pd.read_csv(RUTA_DATOS))
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = cargar_y_dividir()
    print("Entrenamiento:", X_train.shape, "| Evaluación:", X_test.shape)
    print("% impago en entrenamiento:", round((y_train == 0).mean() * 100, 2))
    print("% impago en evaluación:", round((y_test == 0).mean() * 100, 2))

    pre = construir_preprocesador()
    X_train_t = pre.fit_transform(X_train)
    print("Forma transformada:", X_train_t.shape)
    print("Nulos tras transformar:", int(np.isnan(X_train_t).sum()))