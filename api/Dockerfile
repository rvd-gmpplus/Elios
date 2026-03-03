FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

# Context is the 'api/' folder, so copy within this dir
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the whole app folder (the current context)
COPY . /app/api/

ENV PORT=8000
CMD ["bash", "-lc", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
