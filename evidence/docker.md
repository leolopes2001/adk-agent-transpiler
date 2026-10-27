# Imagens Docker: tamanho e tempo de subida

Medido em 2026-10-27 num MacBook Apple Silicon (arm64), Docker Engine 29.4.0 (OrbStack, store de imagens do containerd) e Docker Compose 5.1.2.

## Tamanho das imagens

Base: `python:3.12.16-slim-trixie`, fixada por digest. Ela ocupa 148 MB descompactada e 44 MB comprimida. Cada imagem instala só o extra do seu serviço.

| Imagem | Raiz descompactada | Comprimida | `docker images` | Referência |
|---|---|---|---|---|
| `lab-agent-scheduler-api` | 250 MB | 73 MB | 343 MB | < 250 MB |
| `lab-agent-rag-mcp` | 262 MB | 78 MB | 360 MB | < 250 MB |
| `lab-agent-ocr-mcp` | 466 MB | 169 MB | 667 MB | < 450 MB |
| `lab-agent-agent` | 277 MB | 79 MB | 376 MB | < 450 MB |

Como cada coluna foi medida:

- **Raiz descompactada:** `du -sxm /` dentro de um container da imagem.
- **Comprimida:** `docker image inspect --format '{{.Size}}'`, o tamanho que é baixado de um registry.
- **`docker images`:** soma o conteúdo comprimido e o descompactado (store do containerd), por isso fica maior que as outras duas.

O que mais pesa:

- **`ocr-mcp`:** o Tesseract com os dados de português (apt, 109 MB), o OpenCV headless (87 MB com as bibliotecas) e o numpy (67 MB).
- **`rag-mcp`:** o numpy, que vem do `rank-bm25` (67 MB de 115 MB do venv).
- **Todas as imagens:** o bytecode é pré-compilado no build (`UV_COMPILE_BYTECODE=1`). Com a raiz read-only, o Python não consegue gravar `.pyc` em tempo de execução e recompilaria tudo a cada partida.

Para RAG e OCR, a referência é ultrapassada em 12 e 16 MB. A diferença fica no tamanho descompactado; o comprimido fica abaixo de 170 MB em todas as imagens.

## Tempo de subida

| Cenário | Comando | Tempo |
|---|---|---|
| Frio | `docker compose --profile cli build --no-cache` e depois `docker compose up -d --wait` | build 120 s + up 12 s = **132 s** |
| Quente | `make up` depois de `make down` (imagens prontas, banco recriado) | **15 s** |

- **Frio:**
  - Começa sem imagens do projeto, sem containers e sem volumes.
  - As imagens base já estavam no disco, e o cache de download do uv (`--mount=type=cache`) não é apagado pelo `--no-cache`.
  - Numa máquina sem nada disso, some o download das bases e dos pacotes.
- **Quente:** o tempo vai até os quatro serviços ficarem healthy. A maior parte é o Postgres iniciando o banco do zero e a API rodando as migrações antes do `/health/ready` responder.

## Arquiteturas

As quatro imagens foram construídas para `linux/arm64` (nativo) e `linux/amd64` (emulado). Na imagem amd64 do OCR, `tesseract --version` (5.5.0) e `import cv2` funcionam. Na do agente, `lab-transpile schema` roda.
