FROM python:3.11-slim AS base

WORKDIR /app

COPY . .
ENV PYTHONPATH=/app

CMD ["python", "-m", "server.main"]
