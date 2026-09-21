# CPU-only image on a small base: about 1.5 GB instead of the 3.4 GB CUDA image, and the base is shared with the other Python
# services. Build context = project root (see docker-compose.yml). GPU users: see Dockerfile.cuda.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/app/models
WORKDIR /app

COPY gnn_aml_core/requirements.txt gnn_aml_core/requirements.txt
# PyTorch CPU wheel first (about 190 MB), then the rest. numpy<2 because torch 2.4.1 was built for NumPy 1.x.
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install "numpy<2" -r gnn_aml_core/requirements.txt

COPY gnn_aml_core gnn_aml_core

EXPOSE 8000
CMD ["uvicorn", "gnn_aml_core.main:app", "--host", "0.0.0.0", "--port", "8000"]
