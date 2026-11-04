# Proyecto Final - Machine Learning 2026

Modelo supervisado predictivo sobre créditos.

## Estructura
- mlops_pipeline/: notebooks y scripts del pipeline
- Base_de_datos.csv: dataset del proyecto

## Cómo ejecutar
1. Ejecutar `setup.bat` para crear el entorno e instalar dependencias.
2. Abrir los notebooks de `mlops_pipeline/`.

## Evaluación de modelos

Se compararon cinco modelos con las métricas de la clase minoritaria (impago = 0): precision, recall, F1 y PR-AUC. No se usa accuracy, porque el 95.3% de los clientes paga a tiempo y un modelo que diga "paga" siempre ya tendría ese valor. Los gráficos comparativos se generan al ejecutar `mlops_pipeline/model_training_evaluation.py`.

**Resultados sin la variable `puntaje` (conjunto de evaluación):**

| Modelo | Precision | Recall | F1 | PR-AUC | F1 validación cruzada | Desv. F1 (CV) | Tiempo (s) |
|---|---|---|---|---|---|---|---|
| **Regresión logística** | 0.080 | 0.618 | 0.142 | 0.171 | 0.137 | 0.007 | 0.07 |
| Árbol de decisión | 0.061 | 0.529 | 0.110 | 0.092 | 0.117 | 0.013 | 0.07 |
| KNN | 0.000 | 0.000 | 0.000 | 0.084 | 0.000 | 0.000 | 0.10 |
| Random Forest | 0.247 | 0.176 | 0.206 | 0.159 | 0.131 | 0.047 | 0.46 |
| Gradient Boosting | 0.112 | 0.176 | 0.137 | 0.162 | 0.134 | 0.012 | 0.95 |

**Selección del modelo:** se sumó el puesto de cada modelo en tres criterios: performance (F1 del impago en validación cruzada), consistency (menor variación entre los pedazos de la validación cruzada) y scalability (menor tiempo de entrenamiento y predicción). El conjunto de evaluación no interviene en la elección. Ganó la **regresión logística**, que además tiene el mejor recall y el mejor PR-AUC.

**Variable `puntaje`:** se excluyó del modelo final porque separa perfectamente a los dos grupos (el máximo de quienes no pagaron es 62.67 y el mínimo de quienes sí pagaron es 63.81), lo que sugiere una posible fuga de información. Con ella, los modelos alcanzan métricas cercanas a 1.0, que se consideran poco realistas.
