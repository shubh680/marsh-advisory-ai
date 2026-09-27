# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install essential system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install lightweight CPU-only PyTorch wheel (~200MB vs 2.5GB CUDA bloat)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure media and static directories exist
RUN mkdir -p media/policies media/presentations staticfiles

# Collect static files for WhiteNoise high-performance serving
RUN python manage.py collectstatic --noinput

# Expose Cloud Run port
EXPOSE 8080

# Run migrations on startup and start Gunicorn WSGI server
CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 300 config.wsgi:application"]
