# Build context = project root (see docker-compose.yml).
# Use the same tag you already pulled: docker images | grep pytorch
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/app/models
WORKDIR /app

COPY gnn_aml_core/requirements.txt gnn_aml_core/requirements.txt
RUN pip install -r gnn_aml_core/requirements.txt

COPY gnn_aml_core gnn_aml_core

EXPOSE 8000
CMD ["uvicorn", "gnn_aml_core.main:app", "--host", "0.0.0.0", "--port", "8000"]
