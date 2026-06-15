FROM --platform=linux/amd64 python:3.11.15-slim-bookworm

# Labels
LABEL org.opencontainers.image.title="DocForge"
LABEL org.opencontainers.image.description="Agentic RAG document generator"
LABEL org.opencontainers.image.vendor="Bharat Electronics Limited"
LABEL org.opencontainers.image.version="1.0.0"

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    unzip \
    pkg-config \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    shared-mime-info \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Project directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Install dev tools and UI/testing utilities
RUN uv pip install --system \
    ipython \
    watchdog \
    pytest \
    rich \
    typer \
    textual

# Copy project files
COPY tests .

# Open ports
EXPOSE 8501
EXPOSE 8000
EXPOSE 11434

# Start shell
CMD ["bash"]