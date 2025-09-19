# Multi-stage Dockerfile for pdf2anki

# Build stage
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project metadata and source so editable install can find 'src'
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

# Install Python dependencies and the package (editable)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Runtime stage
FROM python:3.11-slim as runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libfontconfig1 \
    libfreetype6 \
    libjpeg62-turbo \
    libpng16-16 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Add OCR support (optional, controlled by build arg)
ARG ENABLE_OCR=false
RUN if [ "$ENABLE_OCR" = "true" ]; then \
    apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash pdf2anki
USER pdf2anki
WORKDIR /home/pdf2anki

# Copy application code
COPY --chown=pdf2anki:pdf2anki src/ /home/pdf2anki/src/
COPY --chown=pdf2anki:pdf2anki prompts/ /home/pdf2anki/prompts/
COPY --chown=pdf2anki:pdf2anki examples/ /home/pdf2anki/examples/

# Set Python path
ENV PYTHONPATH="/home/pdf2anki/src:$PYTHONPATH"

# Create workspace directory
RUN mkdir -p /home/pdf2anki/workspace/media

# Set default command
ENTRYPOINT ["pdf2anki"]
CMD ["--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pdf2anki version || exit 1

# Labels
LABEL maintainer="Nicholas McCarthy <nicholas@example.com>"
LABEL description="Convert PDF documents to Anki flashcards using LLMs"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/NicholasMcCarthy/pdf2anki"
