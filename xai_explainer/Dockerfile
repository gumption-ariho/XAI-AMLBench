# Build context = project root (see docker-compose.yml).
# Pin VLLM_TAG to the tag you already pulled: docker images | grep vllm
ARG VLLM_TAG=latest
FROM vllm/vllm-openai:${VLLM_TAG}

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# The base image's entrypoint launches vLLM's OpenAI server; we run our own FastAPI app instead.
ENTRYPOINT []

COPY xai_explainer/requirements.txt xai_explainer/requirements.txt
RUN pip install -r xai_explainer/requirements.txt

COPY xai_explainer xai_explainer

EXPOSE 8000
CMD ["uvicorn", "xai_explainer.main:app", "--host", "0.0.0.0", "--port", "8000"]
