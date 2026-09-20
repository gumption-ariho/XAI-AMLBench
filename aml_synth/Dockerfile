# Build context = project root (see docker-compose.yml)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY aml_synth/requirements.txt aml_synth/requirements.txt
RUN pip install -r aml_synth/requirements.txt

COPY aml_synth aml_synth

# Default: write CSVs to /app/data. Override with --to csv,kafka,neo4j
CMD ["python", "-m", "aml_synth.graph_generator"]
