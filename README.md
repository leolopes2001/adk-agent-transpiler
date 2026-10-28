# lab-exam-agent

Transpilador que transforma uma especificação JSON em um agente **Google ADK 2**.

O agente de exemplo lê a foto de um pedido médico, extrai os exames, encontra os códigos no catálogo e registra o agendamento numa API.

## Tecnologias

| Parte | Tecnologias |
|---|---|
| Transpilador | Python 3.12, Pydantic, Jinja2 |
| Agente | Google ADK 2 (Workflow, JoinNode, RequestInput), Gemini |
| OCR | MCP via SSE, Tesseract, OpenCV |
| RAG | MCP via SSE, BM25, RapidFuzz |
| API | FastAPI, PostgreSQL, SQLAlchemy, Alembic |
| Infra | Docker Compose |
| Observabilidade | OpenTelemetry, Aspire Dashboard |
| Qualidade | pytest, Ruff, mypy, import-linter, GitHub Actions |

## Como rodar

### Requisitos

| Item | Versão |
|---|---|
| Docker | Docker Desktop (Windows, macOS) ou Docker Engine (Linux) |
| Docker Compose | 2.24 ou mais novo |
| Chave do Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Python e uv (só para desenvolvimento) | Python 3.12, uv 0.12 ou mais novo |

Para rodar não é preciso Python, uv nem make: só Docker.

### Comandos

```bash
cp .env.example .env         
# preencha GOOGLE_API_KEY no .env

docker compose up -d --build --wait
docker compose run --rm transpile
docker compose run --rm agent run --image /app/inputs/samples/pedido_medico_ficticio.png
```

| Cenário | Comando |
|---|---|
| Revisão humana | `docker compose run --rm agent run --image /app/inputs/samples/pedido_ambiguo.png` |
| Prompt injection | `docker compose run --rm agent run --image /app/inputs/samples/pedido_injecao.png` |
| Gerar a spec alternativa | `docker compose run --rm transpile /app/agent_specs/examples/exam_lookup_only.json` |
| Rodar a spec alternativa | `docker compose run --rm agent run --image /app/inputs/samples/pedido_medico_ficticio.png --agent-dir /app/generated/exam_lookup_only --spec /app/agent_specs/examples/exam_lookup_only.json` |
| Ver o código gerado | `docker compose run --rm export-generated` |
| Swagger | http://localhost:8000/docs (Authorize com `SCHEDULER_API_KEY`) |
| Desligar tudo | `docker compose --profile "*" down -v` |

## Arquitetura

```mermaid
flowchart LR
    IMG[/"imagem"/] --> OCR["ocr<br/>OCR MCP"]
    OCR --> GUARD["guard"]
    subgraph EXT["extraction (paralelo)"]
        LLM["Gemini"] --> JOIN["merge"]
        LEX["busca no catálogo"] --> JOIN
    end
    GUARD --> EXT
    EXT --> RAG["resolve<br/>RAG MCP"]
    RAG --> GATE{"gate"}
    GATE -->|aceito| SCH["schedule<br/>API"]
    GATE -->|dúvida| REV["revisão humana"]
    REV --> SCH
    SCH --> OUT[/"resultado"/]
```

- **A IA interpreta, o código decide.** O Gemini só lê o texto mascarado; quem escolhe o código e agenda é o gate.
- **OCR e RAG são servidores MCP via SSE**, chamados por nós de código, nunca pelo LLM.
- **Workflow em grafo (ADK 2):** sequencial, com fan-out/join paralelo e human-in-the-loop. Equivale a `SequentialAgent` + `ParallelAgent` do ADK 1.

### Estrutura de pastas

```text
.github/workflows/       CI: lint, tipos, testes e E2E
src/
├── transpiler/          spec JSON → validação → geração do agente ADK
├── agent_runtime/       nós do workflow, guardrails e CLI
├── libs/
│   ├── pii_guard/       detecção e mascaramento de dados pessoais
│   └── lab_common/      regra do gate, contratos, logs
└── services/
    ├── ocr_mcp/         servidor MCP de OCR
    ├── rag_mcp/         servidor MCP de busca no catálogo
    └── scheduler_api/   API de agendamento (FastAPI)
agent_specs/             specs dos agentes e o schema do formato
data/                    catálogo de exames
samples/                 imagens de pedidos fictícios
tests/                   unit, contract, integration, security, e2e
docs/                    ADRs
docker/                  Dockerfiles
evidence/                prints e logs de execução
```

## Transpilador

```text
JSON → validação → representação interna → geração (Jinja2) → validação do código → pacote
```

- Mostra todos os erros de uma vez, com o caminho exato na spec.
- Recusa agentes inseguros (por exemplo, agendar sem passar pelo gate).
- A mesma spec gera sempre o mesmo código.

| Spec | Gera |
|---|---|
| [`agent_specs/exam_scheduling_agent.json`](agent_specs/exam_scheduling_agent.json) | Fluxo completo, com LLM, revisão e agendamento |
| [`agent_specs/examples/exam_lookup_only.json`](agent_specs/examples/exam_lookup_only.json) | Só identifica os exames, sem LLM nem agendamento |

## API

| Rota | O que faz |
|---|---|
| `POST /v1/appointments` | Registra um agendamento (idempotente) |
| `GET /v1/appointments` | Lista os agendamentos |
| `GET /v1/appointments/{id}` | Consulta um agendamento |
| `GET /v1/exams/{code}` | Consulta um exame do catálogo |

## Segurança

- **Dados pessoais** mascarados no OCR e checados de novo antes do LLM e nos logs. A API só recebe um pseudônimo.
- **Prompt injection** bloqueado antes do LLM, que não tem ferramentas.
- **Códigos de exame** vêm só do catálogo.
- **Containers** sem root, só leitura e numa rede interna sem internet.

## Logs e traces

```bash
docker compose --profile observability up -d --build --wait
```

Abra http://localhost:18888 e procure pelo `run_id` que a CLI mostra no final.

## Testes

```bash
uv sync --all-extras --group dev
uv run pytest -m "unit or contract"
uv run pytest -m "e2e and not llm" tests/e2e
```

Mais de 2.000 testes, com 98% de cobertura.

## Decisões de arquitetura

| ADR | Decisão |
|---|---|
| [0001](docs/adr/0001-workflow-em-grafo-adk.md) | Workflow em grafo; LLM num único nó |
| [0002](docs/adr/0002-codigo-gerado-como-composicao.md) | Código gerado só compõe peças testadas |
| [0003](docs/adr/0003-pii-mascarada-na-origem.md) | PII mascarada no servidor de OCR |
| [0004](docs/adr/0004-mcp-chamado-por-nos-deterministicos.md) | MCP chamado por código, não pelo LLM |
| [0005](docs/adr/0005-rag-lexico-hibrido.md) | Busca léxica híbrida, calibrada |
| [0006](docs/adr/0006-idempotencia-por-chave-deterministica.md) | Idempotência por chave derivada do conteúdo |
| [0007](docs/adr/0007-sse-como-transporte-mcp.md) | SSE como transporte MCP |
| [0008](docs/adr/0008-ocr-local-tesseract.md) | OCR local com Tesseract |
| [0009](docs/adr/0009-adk-api-contract.md) | Contrato com a API do ADK 2.13.0 |

## Evidências

**Transpilador**

![Geração das duas specs](evidence/transpiler/01-generate.png)

![Spec inválida com os erros acumulados](evidence/transpiler/02-validate-errors.png)

**Agente (CLI)**

![Fluxo feliz](evidence/cli/01-feliz.png)

![Replay idempotente](evidence/cli/02-replay.png)

![Revisão humana do pedido ambíguo](evidence/cli/03-revisao.png)

![Prompt injection bloqueado](evidence/cli/04-injection.png)

![Spec alternativa](evidence/cli/05-spec-alternativa.png)

**Swagger**

![Endpoints e botão Authorize](evidence/swagger/01-endpoints.png)

![Diálogo Authorize com X-API-Key](evidence/swagger/02-authorize.png)

![POST com o exemplo e Idempotency-Key](evidence/swagger/03-post-example.png)

![Resposta 201](evidence/swagger/04-post-201.png)

![Replay com 200 e Idempotent-Replayed](evidence/swagger/05-replay-200.png)

![Mesma chave com outro corpo: 409](evidence/swagger/06-conflict-409.png)

**Docker e testes**

![Serviços do Compose no ar](evidence/docker/01-ps.png)

![Cobertura dos testes](evidence/tests/01-coverage.png)

## Uso de IA


