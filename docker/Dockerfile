# Handbook RAG chatbot (Streamlit + Ollama on the host or in a second container)
# Build context is the PROJECT ROOT (the folder that contains source_code/ and docker/):
#   docker build -f docker/Dockerfile -t handbook-rag .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 1) CPU-only PyTorch first (the default wheel is huge because it contains CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Python dependencies
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt

# 3) Download the reranker model now, so the container also works without internet
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

# 4) Application code and the handbook PDF
COPY source_code/ /app/

# Where to find Ollama. Default: Ollama running on your computer (Docker Desktop).
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    LLM_MODEL=qwen2.5:1.5b \
    PROMPT=balanced

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
