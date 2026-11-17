# Docker deployment

The chatbot runs as a Streamlit web page inside a container. The language model runs in **Ollama**,
either on your computer (default) or in a second container (optional).

## What you need
1. Docker Desktop (running).
2. Ollama on your computer, with two models:
   ```
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
   ```
   (optional, more accurate but slower: `ollama pull qwen2.5:3b`)

## Option A: Ollama on your computer (recommended)
From the project root (the folder that contains `source_code/` and `docker/`):
```
docker compose -f docker/docker-compose.yml up --build
```
Open http://localhost:8501. The first build takes several minutes (it downloads Python packages and
the reranker model). The first question is slow because the index is built.

## Option B: everything in Docker (Ollama in a container too)
```
docker compose -f docker/docker-compose.yml --profile ollama up -d --build
docker exec handbook-rag-ollama ollama pull qwen2.5:1.5b
docker exec handbook-rag-ollama ollama pull nomic-embed-text
```
Then set `OLLAMA_HOST` to the Ollama container and restart the app:
```
# PowerShell
$env:OLLAMA_HOST="http://ollama:11434"; docker compose -f docker/docker-compose.yml --profile ollama up -d
```

## Volumes (data that survives restarts)
| Volume | Inside the container | Purpose |
|---|---|---|
| `../source_code/major_handbook_pdfs` | `/app/major_handbook_pdfs` | handbook PDF (put other PDFs here) |
| `chroma_db` | `/app/chroma_db` | vector database (index) |
| `uploads` | `/app/uploads` | PDFs uploaded in the web page |
| `ollama_models` | `/root/.ollama` | models (Option B only) |

## Useful commands
```
docker compose -f docker/docker-compose.yml logs -f app     # see logs
docker compose -f docker/docker-compose.yml down            # stop
docker compose -f docker/docker-compose.yml down -v         # stop and delete the index/uploads
```

## Troubleshooting
- **"Connection refused" / app cannot reach Ollama (Option A):** check that Ollama is running
  (`ollama list`). If it still fails, set the environment variable `OLLAMA_HOST=0.0.0.0` on your
  computer, restart Ollama, and run again.
- **Port 8501 is already used:** stop the local `streamlit run app.py` first.
- **Changed the PDF or the code:** run `docker compose ... up --build`. To rebuild the index,
  delete the `chroma_db` volume (`down -v`).
