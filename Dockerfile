FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api/ /app/api/
COPY src/ /app/src/

ENV PORT=8000
CMD ["bash", "-lc", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
