"""model_training_evaluation.py
Entrena y evalúa varios modelos y selecciona el mejor.
"""
import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ft_engineering import cargar_y_dividir, construir_preprocesador


def definir_modelos():
    """Modelos a comparar. class_weight='balanced' compensa el 4.7% de impago
    (KNN no tiene ese parámetro)."""
    return {
        "Regresión logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Árbol de decisión": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight="balanced",
            random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42),
    }


def build_model(modelo, incluir_puntaje=True):
    """Pipeline completo: transformaciones de ft_engineering + modelo."""
    return Pipeline([
        ("preprocesador", construir_preprocesador(incluir_puntaje)),
        ("modelo", modelo),
    ])

def summarize_classification(y_true, y_pred, y_proba):

    return {
        "precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=0),
        "f1": f1_score(y_true, y_pred, pos_label=0),
        "pr_auc": average_precision_score(y_true == 0, 1 - y_proba),
    }


def evaluar_modelos(X_train, X_test, y_train, y_test, incluir_puntaje):
    """Entrena cada modelo y mide desempeño, consistencia (validación cruzada)
    y escalabilidad (tiempo de entrenamiento + predicción)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_impago = make_scorer(f1_score, pos_label=0)
    filas, pipelines = [], {}

    for nombre, modelo in definir_modelos().items():
        pipe = build_model(modelo, incluir_puntaje)

        inicio = time.perf_counter()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        tiempo = time.perf_counter() - inicio

        f1_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=f1_impago)

        fila = {"modelo": nombre, "con_puntaje": incluir_puntaje}
        fila.update(summarize_classification(y_test, pred, proba))
        fila.update({"f1_cv_media": f1_cv.mean(), "f1_cv_std": f1_cv.std(),
                     "tiempo_s": tiempo})
        filas.append(fila)
        pipelines[nombre] = pipe

    return pd.DataFrame(filas), pipelines


def seleccionar_mejor(tabla):
    """Suma el puesto de cada modelo en desempeño (F1), consistencia (menor
    desviación en validación cruzada) y escalabilidad (menor tiempo).
    Gana la suma más baja; el empate lo decide el F1."""
    tabla = tabla.copy()
    tabla["ranking"] = (
        tabla["f1"].rank(ascending=False)
        + tabla["f1_cv_std"].rank(ascending=True)
        + tabla["tiempo_s"].rank(ascending=True)
    )
    mejor = tabla.sort_values(["ranking", "f1"], ascending=[True, False]).iloc[0]
    return mejor["modelo"], tabla


def entrenar_y_seleccionar(datos, incluir_puntaje):
    """Devuelve el nombre, el objeto del mejor modelo y la tabla de resultados."""
    tabla, pipelines = evaluar_modelos(*datos, incluir_puntaje)
    nombre, tabla = seleccionar_mejor(tabla)
    return nombre, pipelines[nombre], tabla


def graficar(resultados):
    """Gráficos comparativos de los modelos."""
    datos = resultados.melt(
        id_vars=["modelo", "con_puntaje"],
        value_vars=["precision", "recall", "f1", "pr_auc"],
        var_name="métrica", value_name="valor")
    g = sns.catplot(data=datos, x="métrica", y="valor", hue="modelo",
                    col="con_puntaje", kind="bar", height=4, aspect=1.2)
    g.figure.suptitle("Métricas en evaluación (clase de interés: impago)", y=1.03)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    titulos = {
        "f1_cv_std": "Consistencia: variación del F1 entre folds (menor es mejor)",
        "tiempo_s": "Escalabilidad: segundos de entrenamiento + predicción",
    }
    for ax, (columna, titulo) in zip(axes, titulos.items()):
        sns.barplot(data=resultados, x="modelo", y=columna, hue="con_puntaje", ax=ax)
        ax.set_title(titulo, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    datos = cargar_y_dividir()
    tablas, mejores = [], {}

    for incluir_puntaje in (True, False):
        nombre, modelo, tabla = entrenar_y_seleccionar(datos, incluir_puntaje)
        mejores[incluir_puntaje] = (nombre, modelo)
        tablas.append(tabla)

    resultados = pd.concat(tablas, ignore_index=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(resultados.round(3).to_string(index=False))
    print()
    print("Mejor modelo CON puntaje (solo comparación):", mejores[True][0])
    print("MODELO SELECCIONADO (sin puntaje):", mejores[False][0])

    
modelo_final = mejores[False][1]

# Guardar el modelo seleccionado junto con su preprocesamiento
ruta_modelo = Path(__file__).resolve().parent / "modelo_final.joblib"
joblib.dump(modelo_final, ruta_modelo)

print(f"\nModelo guardado correctamente en: {ruta_modelo}")

graficar(resultados)
