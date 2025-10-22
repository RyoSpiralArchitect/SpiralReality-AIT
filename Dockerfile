FROM python:3.11-slim AS base

WORKDIR /app

COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
