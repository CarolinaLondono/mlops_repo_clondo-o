
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY mlops_pipeline/ ./mlops_pipeline/

EXPOSE 8000

CMD ["uvicorn", "mlops_pipeline.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
