FROM python:3.12-slim

WORKDIR /app

# System deps for chromadb/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install roampal from PyPI (same as users get)
RUN pip install --no-cache-dir roampal

# Data volume — mount here to persist memories across restarts
ENV ROAMPAL_DATA_PATH=/data
VOLUME /data

# Default port (prod)
EXPOSE 27182

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:27182/api/health')" || exit 1

# Bind to 0.0.0.0 so the port is accessible outside the container
CMD ["python", "-m", "uvicorn", "roampal.server.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "27182"]
