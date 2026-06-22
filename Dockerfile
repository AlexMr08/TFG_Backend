FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by some Python packages (Pillow, Postgres client, build tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        postgresql-client \
        libpq-dev \
        libjpeg-dev \
        zlib1g-dev \
        libgl1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy only application code and necessary scripts (avoid large datasets/models)
# `app/` already contains `app/clases/`, so copying `clases/` separately fails
# when the build context doesn't have a top-level `clases/` directory.
COPY app/ ./app/
COPY ingestSQL.py ./ingestSQL.py
COPY run_login_stress.ps1 ./run_login_stress.ps1
COPY requirements.txt /app/requirements.txt

# Create a non-root user and switch
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Default command (in dev you can pass --reload; in prod drop it)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
