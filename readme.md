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

## Ingeniería de características

El procesamiento de datos se realiza mediante `mlops_pipeline/ft_engineering.py`, donde se prepara la información para el entrenamiento y evaluación de los modelos supervisados.

Este proceso incluye limpieza de datos, selección de características, transformaciones y separación de los conjuntos de entrenamiento y evaluación.

El modelo seleccionado se guarda en `mlops_pipeline/modelo_final.joblib` para reutilizarlo durante el despliegue sin necesidad de entrenarlo nuevamente.

## Monitoreo de datos

Se implementó una aplicación en Streamlit para identificar posibles cambios en las distribuciones de los datos mediante la prueba estadística Kolmogorov-Smirnov (KS).

La aplicación permite:

- Comparar distribuciones históricas y actuales.
- Visualizar los resultados estadísticos por variable.
- Identificar cambios mediante indicadores y alertas.
- Cargar archivos CSV para analizar nuevos datos.

Para ejecutar la aplicación:

```bash
python -m streamlit run mlops_pipeline/model_monitoring.py
```

Durante las pruebas se simuló un aumento del 50 % en el capital prestado. El sistema identificó una diferencia estadísticamente significativa y activó la alerta correspondiente.

En ausencia de un archivo externo, se utilizan datos de evaluación como demostración del monitoreo.

## API de predicción

Se desarrolló una API REST utilizando FastAPI para disponibilizar el modelo entrenado mediante el endpoint `/predict`.

La API admite solicitudes en formato JSON o CSV y permite realizar predicciones individuales o por lotes.

Para ejecutarla localmente desde la raíz del proyecto:

```bash
python -m uvicorn mlops_pipeline.model_deploy:app --reload
```

Documentación interactiva:

`http://localhost:8000/docs`

La respuesta incluye la clasificación del comportamiento de pago y la probabilidad estimada de impago.

## Despliegue con Docker

Se creó una imagen Docker que contiene la API, el modelo entrenado y sus dependencias.

Para construir la imagen:

```bash
docker build -t mlops-creditos .
```

Para ejecutar el contenedor:

```bash
docker run --rm -d --name mlops-api -p 8001:8000 mlops-creditos
```

Documentación de la API dentro del contenedor:

`http://localhost:8001/docs`

Se verificó el funcionamiento del endpoint `/predict` con dos registros en formato CSV, obteniendo las mismas predicciones que en la ejecución local.

Para detener el contenedor:

```bash
docker stop mlops-api
```

## Análisis de calidad del código

El repositorio está vinculado con SonarQube Cloud mediante GitHub Actions.

El archivo `.github/workflows/build.yml` configura el análisis sobre la rama `master` y las solicitudes de integración. El archivo `sonar-project.properties` contiene la identificación del proyecto y la configuración del análisis.

El propósito de esta integración es revisar la mantenibilidad, seguridad y calidad del código fuente.