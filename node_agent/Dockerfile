# Build context = ./node_agent
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY agent.py .

EXPOSE 9000
CMD ["python", "agent.py"]
