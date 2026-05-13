# Deployment

This project runs as three Docker Compose services:

- `db`: Postgres with pgvector
- `api`: FastAPI backend
- `streamlit`: web UI

Ollama runs directly on the host server, outside Docker. This keeps GPU/MIG
configuration simpler and lets the app use the server's existing Ollama install.

## Server requirements

- Docker Engine with Compose plugin
- Ollama installed on the host server
- NVIDIA GPU driver, if you want host Ollama to use GPU acceleration

## Host Ollama

Make sure host Ollama has the configured models:

```bash
ollama pull qwen2.5:3b
```

The app uses `qwen2.5:3b` for both SQL-agent reasoning and response rewriting
by default, because it fits smaller server memory limits.

Ollama must listen on an address containers can reach. For a systemd install,
create an override:

```bash
sudo systemctl edit ollama
```

Add:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Then restart Ollama:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## Deploy

From the project directory on the server:

```bash
docker compose up -d --build
```

The app will be available on:

```text
http://SERVER_IP:8501
```

FastAPI and Postgres are intentionally kept private inside Docker networking.
Streamlit reaches FastAPI through the internal `api:8000` service name. FastAPI
reaches host Ollama through `host.docker.internal:11434`.

## GPU check

On the server, verify the host can see the NVIDIA GPU:

```bash
nvidia-smi
```

Then run a model through host Ollama and watch GPU usage:

```bash
ollama run qwen2.5:3b "Explain SQL joins briefly."
watch -n 1 nvidia-smi
```

## Logs

```bash
docker compose logs -f api
docker compose logs -f streamlit
```
