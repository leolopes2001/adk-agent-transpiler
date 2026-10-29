# ADK Agent Transpiler

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

### Passo a passo

```bash
# 1. Configurar (preencha GOOGLE_API_KEY no .env)
cp .env.example .env

# 2. Subir os serviços
docker compose up -d --build --wait

# 3. Gerar o agente a partir da spec
docker compose run --rm transpile /app/agent_specs/exam_scheduling_agent.json

# 4. Rodar o agente com um pedido médico
docker compose run --rm agent run --image /app/inputs/samples/pedido_medico_ficticio.png

# 5. Desligar tudo
docker compose --profile "*" down -v
```

### Outros cenários

| Cenário | Comando |
|---|---|
| Revisão humana | `docker compose run --rm agent run --image /app/inputs/samples/pedido_ambiguo.png` |
| Prompt injection | `docker compose run --rm agent run --image /app/inputs/samples/pedido_injecao.png` |
| Swagger | http://localhost:8000/docs (Authorize com `SCHEDULER_API_KEY`) |

### Rodar outro agente

Cada spec gera um agente, escolhido pelo nome com `--agent`:

```bash
docker compose run --rm transpile /app/agent_specs/examples/exam_lookup_only.json
docker compose run --rm agent run --agent exam_lookup_only --image /app/inputs/samples/pedido_medico_ficticio.png
docker compose run --rm agent list          # lista os agentes gerados
docker compose run --rm export-generated    # copia o código gerado para ./generated/
```

## Arquitetura

![Arquitetura](evidence/architecture/arquitetura.svg)

![Fluxo do agente](evidence/architecture/fluxo.svg)

- **A IA interpreta, o código decide.** O Gemini só lê o texto mascarado; quem escolhe o código e agenda é o gate.
- **OCR e RAG são servidores MCP via SSE**, chamados por nós de código, nunca pelo LLM.
- **O `rag-mcp` faz a etapa de recuperação** (busca léxica no catálogo); os candidatos não são entregues ao LLM, de propósito: quem decide é o gate.
- **Modelo `gemini-flash-latest`:** o apelido evita quebra quando versões antigas forem aposentadas; em produção, a versão seria fixada e trocada só depois de validar com testes.
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
| [`exam_scheduling_agent.json`](agent_specs/exam_scheduling_agent.json) | Fluxo completo: LLM, revisão humana e agendamento |
| [`examples/exam_lookup_only.json`](agent_specs/examples/exam_lookup_only.json) | Só identifica os exames, sem LLM nem agendamento |
| [`examples/exam_validation_errors.json`](agent_specs/examples/exam_validation_errors.json) | Spec inválida de propósito, para ver os erros |

```bash
# Gerar outro agente e rodá-lo pelo nome
docker compose run --rm transpile /app/agent_specs/examples/exam_lookup_only.json
docker compose run --rm agent run --agent exam_lookup_only --image /app/inputs/samples/pedido_medico_ficticio.png

# Ver os erros de uma spec inválida, sem gerar nada
docker compose run --rm --entrypoint lab-transpile transpile validate /app/agent_specs/examples/exam_validation_errors.json
```

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

### Dados pessoais

Tudo roda localmente; o único serviço externo é o Gemini, e não há embedding externo. As camadas, em ordem:

1. **Regras** (`pii_guard`): CPF, CNS e RG com dígito verificador e tolerância a erro de OCR, telefone, e-mail, CEP, endereço, data de nascimento, CRM, carteirinha, qualquer valor depois de um rótulo de dado pessoal e nomes depois de rótulos como "Paciente" ou "Dr.".
2. **NER local** (`pii_guard.ner`): Presidio com spaCy `pt_core_news_md`, só para nomes de pessoa. Lê o texto inteiro e cada linha, no original e numa cópia com cada palavra capitalizada (pega nomes em MAIÚSCULAS). Seus achados perdem qualquer sobreposição com as regras e são cortados nas palavras do catálogo, nos termos médicos (`medical_terms.txt`, com tolerância a erro de OCR), nos rótulos e em palavras com dígito, para nunca mascarar um exame.
3. **Minimização** (nó `llm_extract`): o LLM recebe só linhas candidatas a exame: itens de lista, títulos de seção curtos e linhas parecidas com o catálogo. Linha com rótulo de dado pessoal nunca entra, nem mascarada.
4. **Tripwire** antes do LLM: roda os mesmos detectores, NER incluído, sobre o trecho do documento; qualquer achado bloqueia a chamada (`PII_TRIPWIRE`), sem enviar o texto original.

Regras e NER rodam no servidor OCR (que nunca devolve texto sem mascarar) e de novo no agente (nó `guard` e tripwire). Se o modelo não carregar ou falhar, nada sai: o OCR responde `PII_ENGINE_FAILURE` e o tripwire bloqueia. O mascaramento é irreversível (`[PESSOA_1]`, `[CPF_1]`); não existe mapa para recuperar os valores.

**Configuração:**

- O modelo é uma dependência fixada (extra `ner` no `pyproject.toml` e `uv.lock`), instalada no build das imagens do OCR e do agente; nunca é baixado em execução.
- A minimização tem três parâmetros no nó `llm_extract` da spec: `min_catalog_similarity` (0,7), `max_title_chars` (40) e `min_kept_lines` (1). Com menos linhas candidatas que `min_kept_lines`, o LLM recebe o texto inteiro já mascarado pelo `guard`, que ainda passa pelo tripwire.
- Um exame lido como nome se resolve adicionando a palavra a `src/libs/pii_guard/medical_terms.txt`.

**Medição** (`uv run python scripts/measure_pii.py`): 58 trechos sintéticos em `tests/data/pii_ner_cases.jsonl`, viram testes em `tests/unit/pii/test_ner.py`; a varredura do catálogo com o NER fica em `tests/security/test_catalog_ner.py`. "Catálogo mascarado" conta os 310 nomes e aliases do catálogo, como escritos e em MAIÚSCULAS, em 5 contextos.

| Configuração | Nomes /27 | CPF /7 | E-mail /2 | Telefone /3 | Endereço /2 | Cidade sem rótulo /2 | Termos médicos mascarados | Catálogo mascarado | ms/trecho |
|---|---|---|---|---|---|---|---|---|---|
| Regras | 8 | 7 | 2 | 3 | 2 | 0 | 0 | 0/3100 | 0,1 |
| Regras + NER no texto original | 21 | 7 | 2 | 3 | 2 | 0 | 0 | 0/3100 | 3,0 |
| **Regras + NER no original e na cópia capitalizada (em uso)** | **25** | 7 | 2 | 3 | 2 | 0 | 0 | 0/3100 | 5,2 |

Os 2 nomes que faltam são escapes conhecidos, listados abaixo; o corpus os marca com `escapes`, e um teste falha quando um deles passa a ser mascarado, para que esta tabela seja atualizada.

LOCATION (cidades e bairros) ficou de fora: achou 1 de 2 cidades, tomou "FERRLTINA" por lugar e, sem prioridade menor, quebrava endereços que as regras já pegam.

**Custo** (medido nesta máquina; tamanhos de `docker images`):

| | Antes (só regras) | Depois (regras + NER) |
|---|---|---|
| Imagem `lab-agent-ocr-mcp` | 681 MB | 1,11 GB |
| Imagem `lab-agent-agent` | 382 MB | 878 MB |
| Carga do modelo | — | ~1,4 s no container do OCR |
| Mascarar um pedido das amostras (15 a 21 linhas) | ~1 ms | 80 a 125 ms |

O aumento das imagens vem do spaCy (133 MB), do modelo (52 MB), do numpy e do phonenumbers do Presidio. O OCR carrega o modelo na subida, antes de o `/health` responder. O agente carrega na primeira execução que precisa dele (`guard` ou tripwire), uma vez por processo; `lab-agent list`, `--help` e erros de argumento não o carregam. Se a carga falhar, a execução para no `guard` com `PII_ENGINE_FAILURE`, antes de qualquer envio ao LLM.

**O que continua escapando** (nenhuma camada promete 100%):

- Cidade ou bairro sem rótulo ("Reside em Contagem").
- Sobrenome igual a um termo médico ou epônimo no fim de um nome sem rótulo: "Ana Ferro - 42 anos" e "Maria Chagas - 30 anos" mascaram só o primeiro nome. Com rótulo ("Paciente: Ana Ferro") o nome inteiro é mascarado.
- Um nome que seja só um termo médico, epônimo ou rótulo: "Paciente: Chagas", "Nascimento".
- Nomes que o OCR deformou a ponto de o modelo não reconhecer.
- O NER também mascara palavras que não são dados pessoais (o nome do laboratório, por exemplo). Nenhum exame do corpus nem do catálogo é mascarado, mas um exame fora do catálogo ainda pode ser lido como nome.
- Os logs usam só as regras de alta precisão: por construção, não recebem texto do documento.

## Limitações

- A calibração (limiares da busca, do gate e da minimização) foi feita com pedidos fictícios gerados para o projeto; com layouts muito diferentes, mais itens vão para a revisão humana ou ficam sem agendar, e nunca são agendados sem passar pelo gate (exemplos em `tests/data/images/pedido_*`).

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

Prints de execução real, gerados com os comandos deste README.

| Área | O que mostra |
|---|---|
| [Transpilador](#transpilador-1) | Geração do agente, erros de validação e lista de agentes |
| [Agente (CLI)](#agente-cli) | Fluxo completo, idempotência, revisão humana, prompt injection e pedido manuscrito |
| [API](#api-1) | Swagger e consulta dos agendamentos registrados |
| [Logs e traces](#logs-e-traces-1) | Logs estruturados e trace de uma execução no Aspire |
| [Docker](#docker) | Serviços do Compose no ar |
| [Testes](#testes-1) | Suíte de testes e cobertura |

### Transpilador

**Geração do agente a partir da spec.** O transpilador lê `exam_scheduling_agent.json` e gera `agent.py`, `__init__.py` e `_manifest.json`.

![Geração do agente exam_scheduling_agent](evidence/transpiler/01-generate.jpeg)

**Spec inválida.** Os quatro erros aparecem de uma vez, cada um com código, caminho na spec e dica de correção. Nenhum arquivo é gerado.

![Spec inválida com os erros acumulados](evidence/transpiler/02-validate-errors.jpeg)

**Agentes gerados.** `agent list` mostra os dois agentes disponíveis para `--agent`.

![Lista dos agentes gerados](evidence/transpiler/03-show-agent-list.png)

### Agente (CLI)

**Fluxo completo.** Pedido digitado: 7 exames agendados, 8 dados pessoais mascarados. "PCR" e "Urina tipo 1 (EAS)" foram resolvidos para os nomes do catálogo.

![Fluxo completo com 7 exames agendados](evidence/cli/01-fluxo-completo.jpeg)

**Idempotência.** A mesma imagem rodada de novo devolve o mesmo `ID` e nada é duplicado.

![Replay idempotente sem duplicar a solicitação](evidence/cli/02-idempotencia.jpeg)

**Revisão humana.** "T4" é ambíguo (T4 livre ou T4 total, ambos com similaridade 0,93). O gate não decide sozinho e pede a escolha ao usuário.

![Revisão humana do pedido ambíguo](evidence/cli/03-revisao-humana.jpeg)

**Prompt injection.** O pedido contém trechos que parecem instruções para a IA. Ele é bloqueado antes do LLM e nada é agendado.

![Prompt injection bloqueado](evidence/cli/04-prompt-injection.jpeg)

**Pedido manuscrito.** Pedido escrito à mão em papel pautado ([`samples/pedido_manuscrito.jpg`](samples/pedido_manuscrito.jpg)). O OCR leu "TSH" como "15H"; a IA sugeriu TSH e a confirmação veio da revisão humana. "FERRLTINA" foi resolvido para Ferritina. 4 exames agendados, 7 dados pessoais mascarados.

![Pedido manuscrito com revisão humana](evidence/cli/05-pedido-manuscrito.jpeg)

**Pedido manuscrito, segunda execução.** Mesmo com o OCR de um manuscrito, a chave de idempotência se mantém: mesmo `ID`, nada duplicado.

![Replay idempotente do pedido manuscrito](evidence/cli/06-pedido-manuscrito-idempotencia.jpeg)

### API

**Swagger.** Rotas de agendamento, catálogo e health, autenticação por `X-API-Key` e contrato sem dados pessoais (só o pseudônimo `patient_ref`).

![Endpoints no Swagger](evidence/swagger/01-endpoints.jpeg)

**Consulta dos agendamentos.** `GET /v1/appointments` devolve a solicitação do pedido manuscrito com status `requested`, os códigos do catálogo e o hash do documento de origem.

![GET /v1/appointments com 200 OK](evidence/swagger/02-test-endpoint.png)

### Logs e traces

**Logs estruturados.** Logs em JSON dos serviços `rag-mcp`, `ocr-mcp` e `scheduler-api` no Aspire Dashboard.

![Logs estruturados no Aspire](evidence/observability/logs.jpeg)

**Trace de uma execução.** Uma execução do agente de ponta a ponta (23 spans, 4 serviços): OCR, guard, extração em paralelo (Gemini e busca léxica), resolve, gate e `POST /v1/appointments`.

![Trace de uma execução no Aspire](evidence/observability/trace.jpeg)

### Docker

**Serviços no ar.** `docker compose ps` com todos os serviços saudáveis. Só a API (8000) e o Aspire (18888) ficam expostos, e só em `127.0.0.1`.

![docker compose ps](evidence/docker/02-containers.png)

![Containers no Docker Desktop](evidence/docker/01-containers.png)

### Testes

**Suíte de testes e cobertura.** 2.102 testes passando, com relatório de cobertura por arquivo.

![Testes e cobertura](evidence/tests/01-test.jpeg)

## Transparência e Uso de IA

### Abordagem de desenvolvimento

Desenvolvi por etapas, com IA como apoio ao planejamento, implementação e elaboração de testes. Revisei as entregas, pedi ajustes e validei manualmente o fluxo completo, registrando evidências de funcionamento.

### Referências técnicas

- [Google ADK](https://adk.dev): Workflows em grafo, agentes, callbacks e retomada de execução.
- [Model Context Protocol](https://modelcontextprotocol.io) e o [SDK Python](https://github.com/modelcontextprotocol/python-sdk): servidores MCP via SSE.
- [FastAPI](https://fastapi.tiangolo.com) e [Pydantic](https://docs.pydantic.dev): contratos e validação de dados.
- [Tesseract](https://tesseract-ocr.github.io): OCR local.
- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/): defesas contra prompt injection e excesso de autonomia.

### Estratégia de orquestração

O agente é um Workflow em grafo do Google ADK 2, com rotas explícitas decididas por código:

- **Sequencial:** OCR → guard → extração → busca no catálogo → gate → agendamento.
- **Paralelo:** o Gemini e a busca no catálogo (RAG) rodam juntos e se encontram num `JoinNode`.
- **Human-in-the-loop:** itens ambíguos pausam o fluxo e retomam com a escolha do operador.

O LLM atua em um único passo, interpretando o pedido, sem ferramentas e com saída estruturada. OCR e RAG são chamados via MCP por passos determinísticos, e as decisões críticas (mascaramento de PII, escolha do código do exame e agendamento) ficam no código, para garantir previsibilidade e segurança.
