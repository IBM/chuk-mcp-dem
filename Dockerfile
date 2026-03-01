# DEM MCP Server Dockerfile
# ===================================
# Multi-stage build for optimal image size
# Includes GDAL/rasterio system dependencies

# Build stage
FROM python:3.14-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies (including GDAL for rasterio)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project configuration
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package with all dependencies
# Use --no-cache to reduce layer size
RUN uv pip install --system --no-cache -e .

# Runtime stage
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
# rasterio wheels bundle GDAL but still need system libexpat1, libxml2, libcurl4
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libexpat1 \
    libxml2 \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/README.md ./
COPY --from=builder /app/pyproject.toml ./

# Create non-root user for security
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app

# Switch to non-root user
USER mcpuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    GDAL_HTTP_TIMEOUT=60 \
    GDAL_HTTP_MAX_RETRY=5 \
    GDAL_HTTP_RETRY_DELAY=2 \
    GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    CPL_VSIL_CURL_CHUNK_SIZE=10485760 \
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); import chuk_mcp_dem; print('OK')" || exit 1

# Default command - run MCP server in HTTP mode for Docker
CMD ["python", "-m", "chuk_mcp_dem.server", "http"]

# Expose port for HTTP mode
EXPOSE 8003

# Labels for metadata
LABEL description="DEM MCP Server - Digital Elevation Model Discovery, Retrieval & Terrain Analysis" \
      version="0.2" \
      org.opencontainers.image.title="DEM MCP Server" \
      org.opencontainers.image.description="MCP server for DEM discovery, retrieval and terrain analysis"
