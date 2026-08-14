# =============================================================================
# AlphaPredict - Stock Market Predictor & Financial Intelligence Agent
# Multi-stage build: compile/install dependencies in a throwaway builder
# stage (keeps build-essential, gcc, etc. out of the final image), then
# copy only the resulting virtualenv + application code into a slim
# runtime image.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# System build dependencies: chromadb's chroma-hnswlib and a few other
# transitive deps may need to compile from source depending on platform/
# wheel availability, so a C/C++ toolchain is required at BUILD time only.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Isolate dependencies in a venv so Stage 2 can copy it wholesale without
# also inheriting the build toolchain.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Bake NLTK corpora into the image at build time (tokenizer, lemmatizer,
# stopwords) rather than fetching them on every container start - avoids
# runtime network dependency and cold-start latency in Phase 1's router.
ENV NLTK_DATA=/opt/venv/nltk_data
RUN python -c "\
import nltk; \
[nltk.download(pkg, download_dir='${NLTK_DATA}') for pkg in \
 ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'stopwords']]"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Run as a non-root user (standard hardening for production containers).
RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Bring in the fully-populated venv (deps + NLTK corpora) from the builder;
# no compiler toolchain, apt cache, or pip cache carries over.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA=/opt/venv/nltk_data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Application code for all five phases.
COPY main.py .
COPY nlp_router.py .
COPY vector_store.py .
COPY quant_baseline.py .
COPY langgraph_agent.py .

# Pre-trained Phase 1 artifacts (Keras classifier + vectorizer + label
# encoder) must exist in the build context at ./artifacts before building
# this image - see nlp_router.py's ARTIFACTS_DIR.
COPY artifacts/ ./artifacts/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-level readiness check against the /health endpoint (Phase 5).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else sys.exit(1)"

# NOTE: GOOGLE_API_KEY must be supplied at runtime, e.g.:
#   docker run -e GOOGLE_API_KEY=... -p 8000:8000 alphapredict

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
