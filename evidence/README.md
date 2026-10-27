# Evidências

Esta pasta reúne os artefatos que mostram cada requisito do desafio funcionando. Quase tudo é gerado por um único comando, rodado no host:

```bash
uv run python scripts/collect_evidence.py      # --skip-coverage, --fake-llm
uv run pytest -m security tests/security/test_e2e_pii.py
```

O script:

- **Transpilador:** roda `lab-transpile` (spec com erros, geração das duas specs, diff entre elas).
- **Artefatos:** exporta o `openapi.json` e refaz a calibração do RAG.
- **Cobertura:** mede a cobertura de todos os testes, menos os e2e.
- **Compose:** sobe a stack como projeto `lab-agent-e2e`, roda os cenários com o Gemini real (se houver `GOOGLE_API_KEY` no ambiente ou no `.env`) e derruba tudo com `down -v`.

Os containers têm sistema de arquivos read-only e não gravam no repositório. Tudo é coletado pelo host: saída da CLI, `docker compose logs`, respostas da API e linhas do banco.

O override `docker-compose.e2e.yml` faz duas coisas: publica a API numa porta aleatória e aponta o agent para o modelo fake. O script desfaz a segunda para os cenários com LLM (`APP_ENV=dev`, `AGENT_LLM_BACKEND=gemini`). As imagens, as redes e o hardening são os do `docker-compose.yml`.

## Índice por requisito

| Requisito | Evidência |
|---|---|
| Transpilador JSON → Python ADK | [`transpiler/generate.txt`](transpiler/generate.txt) (as duas specs geradas). [`transpiler/exam_scheduling_agent.agent.py.txt`](transpiler/exam_scheduling_agent.agent.py.txt) e [`transpiler/exam_lookup_only.agent.py.txt`](transpiler/exam_lookup_only.agent.py.txt): o `agent.py` de cada spec, copiado do volume onde o container o gerou. A extensão `.txt` deixa as cópias fora do lint. |
| Specs diferentes geram código diferente | [`transpiler/diff-specs.diff`](transpiler/diff-specs.diff) e [`transpiler/diff-generated.diff`](transpiler/diff-generated.diff). A spec alternativa não tem LLM, nem revisão, nem agendamento, e o código gerado perde os nós e imports correspondentes. |
| Código gerado usa só o Google ADK para instanciar agentes | Os dois `agent.py`. O agente LLM, o workflow e a app são `Agent`, `Workflow` e `App` do `google.adk`, sem outro framework de agentes. A spec alternativa não tem LLM e por isso não tem `Agent`. |
| Validação de input com mensagens de erro claras | [`transpiler/validate-errors.txt`](transpiler/validate-errors.txt): 6 erros e 2 avisos numa única execução, cada um com código, JSON Pointer, nó e dica. A spec usada é [`transpiler/exam_scheduling_agent_invalid.json`](transpiler/exam_scheduling_agent_invalid.json): a spec principal com uma tool fora da allowlist, um serviço não declarado e uma aresta que chega ao agendamento sem passar pelo gate de confiança. |
| Código gerado executável | [`runs/index.md`](runs/index.md): todos os cenários rodaram o código gerado dentro do container do agent. [`cli/07-spec-alterada.txt`](cli/07-spec-alterada.txt): código desatualizado em relação à spec é recusado com exit 7. |
| OCR como MCP Server via SSE | Em [`logs/compose.log`](logs/compose.log), os eventos `ocr_request`, `ocr_done` e `pii_redacted` do serviço `ocr-mcp`. A rota de falha está em [`cli/05-ilegivel.txt`](cli/05-ilegivel.txt): imagem ilegível, `OCR_LOW_QUALITY` e exit 4. |
| RAG como MCP Server via SSE, catálogo com ≥ 100 exames | O evento `catalog_loaded` do serviço `rag-mcp` em [`logs/compose.log`](logs/compose.log), com a contagem de exames. [`rag_calibration.md`](rag_calibration.md): thresholds do gate calibrados num conjunto de avaliação. A coluna de score das tabelas em [`cli/`](cli/). |
| API de agendamento FastAPI com Swagger que reflete o contrato | [`openapi.json`](openapi.json), igual ao gerado pela app (há teste que compara os dois). [`api/appointment.json`](api/appointment.json): `GET /v1/appointments/{id}` do agendamento do fluxo feliz. Capturas do Swagger: [`screenshots/README.md`](screenshots/README.md). |
| Camada de PII antes do LLM e da persistência | A linha "Dados pessoais mascarados" (contagens por tipo) em [`cli/01-feliz.txt`](cli/01-feliz.txt). [`api/database.txt`](api/database.txt): o banco guarda só o `patient_ref` pseudônimo (HMAC). O teste `tests/security/test_e2e_pii.py` varre logs, saída da CLI, banco e esta pasta. |
| Docker exclusivo + `docker-compose.yml` | [`docker.md`](docker.md): tamanho das imagens, tempos de subida, isolamento. Todas as execuções de [`runs/`](runs/) e [`cli/`](cli/) rodaram no Compose. |
| Fluxo feliz | [`runs/index.md`](runs/index.md) (relatório JSON) e [`cli/01-feliz.txt`](cli/01-feliz.txt): 7 exames, os códigos de `samples/expected/pedido_medico_ficticio.json`, `scheduled`. |
| Idempotência | [`cli/02-replay.txt`](cli/02-replay.txt) e o relatório "Reexecução" em [`runs/index.md`](runs/index.md): `replayed`, mesmo `appointment_id`. [`api/database.txt`](api/database.txt): uma linha por documento. |
| Injeção de prompt | [`cli/03-injection.txt`](cli/03-injection.txt) e o relatório correspondente: `INPUT_BLOCKED`, exit 3, caminho `ocr → guard → abort` (nenhuma chamada ao LLM nem à API). |
| Revisão humana de itens ambíguos | [`cli/04-ambiguo.txt`](cli/04-ambiguo.txt): em modo não interativo o item ambíguo é rejeitado (`needs_human_review`) e o restante é agendado. O modo interativo está nas capturas opcionais. |
| Spec alternativa sem agendamento | [`cli/06-spec-alternativa.txt`](cli/06-spec-alternativa.txt): `exam_lookup_only`, `resolved`, sem chamada à API. |
| Observabilidade (logs estruturados, correlação) | [`logs/compose.log`](logs/compose.log): JSON por serviço, já redigido na origem. O `run_id` da CLI aparece como `request_id` na API. As linhas `agent-run-NN` são o stderr de cada execução do agent; os containers do agent são removidos após cada execução. |
| Testes e cobertura | [`coverage.txt`](coverage.txt): todos os testes, menos os e2e (unit, contract, integration, security). A suíte e2e roda com `uv run pytest -m "e2e and not llm" tests/e2e tests/security/test_e2e_pii.py` e no job `e2e` do CI. |
| README, JSON de exemplo e imagem de teste | `specs/exam_scheduling_agent.json`, `specs/examples/exam_lookup_only.json`, `samples/*.png` e `samples/expected/`. O README da raiz é o da fase final. |

## Conteúdo

| Caminho | Origem |
|---|---|
| `runs/<run_id>.json`, `runs/index.md` | Relatórios JSON dos cenários feliz, replay, injeção e ambíguo, e um índice com LLM, exit e outcome de cada execução. Cada relatório é da mesma execução que a captura correspondente em `cli/` (o `run_id` aparece na última linha da captura). Gerados pelo script. |
| `cli/NN-*.txt` | Saída da CLI em tabela: o comando dentro do container, o stdout, as mensagens `lab-agent:` do stderr e o exit code. Cada cenário roda uma vez; o script lê o relatório JSON dessa mesma execução por um wrapper da CLI. Gerados pelo script. |
| `logs/compose.log` | `docker compose logs` de todos os serviços e o stderr (JSON) de cada execução do agent. Gerado pelo script. |
| `api/appointment.json`, `api/database.txt` | Resposta da API e linhas do banco ao fim da coleta. Gerados pelo script. |
| `transpiler/*` | Saída do `lab-transpile` e cópias do código gerado. Gerados pelo script. |
| `openapi.json` | `uv run python -m scheduler_api.openapi evidence/openapi.json`, também chamado pelo script. |
| `rag_calibration.md` | `scripts/calibrate_rag.py`, também chamado pelo script. É determinístico. |
| `coverage.txt` | Gerado pelo script, a menos que se use `--skip-coverage`. |
| `docker.md` | Medições feitas à mão na fase de Docker. |
| `screenshots/` | Capturas manuais; a lista está em [`screenshots/README.md`](screenshots/README.md). |
| `e2e/` | Logs salvos pelos testes e2e quando um teste falha ou com `KEEP_EVIDENCE=1`. Ignorado pelo git; no CI vira artefato. |

## Sem dados pessoais

As amostras são fictícias, mas são tratadas como dados reais. Nenhum arquivo desta pasta deve conter nome, CPF (com ou sem máscara), telefone, data de nascimento, CNS, CRM ou endereço das amostras. O teste `tests/security/test_e2e_pii.py::test_no_personal_data_in_the_evidence_folder` procura esses valores e os padrões genéricos (CPF com dígito verificador válido, CNS, e-mail, telefone) em todos os arquivos de texto. Ele também roda no CI. As imagens de `screenshots/` são revisadas à mão.
