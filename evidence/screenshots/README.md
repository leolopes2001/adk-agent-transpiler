# Capturas de tela

As capturas são tiradas à mão; nada aqui é gerado por script. Salve cada imagem nesta pasta com o nome indicado.

## Preparação

```bash
cp .env.example .env            # e preencha GOOGLE_API_KEY
docker compose up -d --build --wait
```

Abra `http://localhost:8000/docs`. A chave da API é o valor de `SCHEDULER_API_KEY` no `.env`.

Use apenas os exemplos do próprio Swagger. Eles usam um `patient_ref` pseudônimo e nenhum dado pessoal. Antes de salvar, confira que a imagem não mostra a chave da API nem o conteúdo do `.env`.

## Swagger

| Arquivo | O que mostrar |
|---|---|
| `swagger-01-endpoints.png` | A página inicial com a lista de endpoints (`/v1/appointments`, `/v1/exams/{code}`, `/health/*`) e o botão **Authorize**. |
| `swagger-02-authorize.png` | O diálogo **Authorize** com o esquema `X-API-Key` (header). Tire o print antes de colar a chave, ou com o campo mascarado. |
| `swagger-03-post-example.png` | `POST /v1/appointments` em **Try it out**, com o exemplo "Two catalog exams" selecionado e um `Idempotency-Key`, por exemplo `demo-key-0001`. |
| `swagger-04-post-201.png` | A resposta **201** dessa requisição: corpo com `appointment_id`, `status: requested` e `patient_ref`, e os headers `Location` e `X-Request-ID`. |
| `swagger-05-replay-200.png` | A mesma requisição de novo, com a mesma chave e o mesmo corpo: resposta **200**, header `Idempotent-Replayed: true` e o mesmo `appointment_id`. |
| `swagger-06-conflict-409.png` | A mesma chave com outro corpo (por exemplo, remova um exame): resposta **409**, problem `idempotency_key_reuse`. |

## Opcionais

| Arquivo | O que mostrar |
|---|---|
| `swagger-07-unknown-code-422.png` | O exemplo "Unknown code (422)" com outra chave: resposta **422** `unknown_exam_code`. |
| `cli-01-run.png` | O terminal com `docker compose run --rm transpile` e `docker compose run --rm agent run --image /app/inputs/samples/pedido_medico_ficticio.png`: a tabela de exames e a linha "Agendamento: solicitação registrada". |
| `cli-02-review.png` | `docker compose run --rm agent run --image /app/inputs/samples/pedido_ambiguo.png`: a revisão humana com as opções de T4. |

## Logs e traces (opcional)

Suba a stack com o profile `observability` e rode o fluxo feliz:

```bash
docker compose --profile observability up -d --build --wait
docker compose run --rm transpile
docker compose run --rm agent run --image /app/inputs/samples/pedido_medico_ficticio.png
```

Abra `http://localhost:18888` e anote o `run_id` da última linha da CLI (`Execução <run_id>`).

| Arquivo | O que mostrar |
|---|---|
| `obs-01-traces.png` | A página *Traces* com o trace `lab-agent run` da execução, com o recurso `agent-<run_id>`, os quatro serviços na coluna de recursos e a duração. |
| `obs-02-trace.png` | Esse trace aberto: a linha do tempo com `lab-agent run`, os `invoke_node`, `generate_content`, `mcp ocr.extract_text` com o `tools/call extract_text` do `ocr-mcp` dentro, e o `POST /v1/appointments` com o span da `scheduler-api`. |
| `obs-03-logs-run-id.png` | A página *Structured logs* com o recurso em "(All)" e o `run_id` no filtro: linhas de `agent`, `ocr-mcp`, `rag-mcp` e `scheduler-api`, e a coluna *Trace* com o link para o trace. |

As capturas mostram só ids, códigos de exame, contagens e tempos. Mesmo assim, confira antes de salvar que nenhum detalhe de log aberto mostra algo além disso.

## Depois

```bash
docker compose --profile "*" down -v
```

O teste `tests/security/test_e2e_pii.py` varre os arquivos de texto de `evidence/`, mas não lê imagens. A revisão das capturas é manual.
