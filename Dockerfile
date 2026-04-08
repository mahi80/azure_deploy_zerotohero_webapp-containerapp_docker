# ══════════════════════════════════════════════════════════════════
# Stage 1: builder — install dependencies into a virtual environment
# ══════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

# System packages needed to compile psycopg2 and lightgbm C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first — Docker caches this layer unless requirements.txt changes
COPY requirements.txt .

# Install into an isolated venv for clean copying to runtime stage
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ══════════════════════════════════════════════════════════════════
# Stage 2: runtime — minimal image, no build tools
# ══════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# libpq is needed at runtime for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security — never run as root in production
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy venv from builder (no pip/build tools in final image)
COPY --from=builder /opt/venv /opt/venv

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY app/       ./app/
COPY models/    ./models/

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# ── Health check — Docker / Azure App Service / K8s liveness probe ────────────
# Checks every 30s. Marks container unhealthy if /health returns non-200 for 3 tries.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ──────────────────────────────────────────────────────────────
# workers=1 for single container; increase or use gunicorn for multi-worker
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
