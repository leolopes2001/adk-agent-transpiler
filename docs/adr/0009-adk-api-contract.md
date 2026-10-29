# ADR-0009 — Contrato com a API do google-adk 2.13.0

- **Status:** Aceito
- **Data:** 2026-10-27

## Contexto

O runtime depende de comportamentos do ADK 2 que a documentação ainda não
estabiliza: assinatura de nós, roteamento, `JoinNode`, workflow aninhado,
`Agent` em modo `single_turn`, retry, HITL e MCP. Antes de implementar os nós,
cada suposição virou um teste em `tests/contract/test_adk_contract.py`,
rodando contra a versão instalada (`google-adk` 2.13.0, `mcp` 1.30.0,
`google-genai` 2.33.0, fixadas no `uv.lock`; o `google-adk` também no
`pyproject.toml`). O primeiro teste falha se a
versão mudar, para obrigar a refazer esta verificação.

## Decisão

Confirmado sem ajuste (o código gerado e o template não mudam):

- `FunctionNode(func, name, retry_config, timeout, rerun_on_resume)`; função
  `async (ctx, node_input) -> Event`; `ctx.state` vira `state_delta`.
- Arestas `("START", n)` e `(n, {"rota": m})`, com o mesmo destino em duas rotas.
- `JoinNode` entrega um dict por nome do predecessor, incluindo o `Agent`.
- `Workflow` aninhado como nó; a saída é a do nó terminal interno.
- `Agent(mode="single_turn", output_schema=...)` envia só a entrada do nó como
  conteúdo `user` e a instrução como `system_instruction`; a saída chega como
  dict validado. `HttpRetryOptions` viaja em `config.http_options`.
- `RequestInput` interrompe; o resume via `Runner.run_async(invocation_id=...)`
  com `FunctionResponse` reexecuta o nó com `ctx.resume_inputs[interrupt_id]`.
- `RetryConfig(exceptions=[...])` filtra pela MRO da exceção.

Diferenças e o que fazemos:

1. **Primeira entrada é `types.Content`.** O nó `ocr_extract` lê o JSON de
   `RunInput` do texto da mensagem.
2. **`Event(output=BaseModel)` chega ao próximo nó como a própria instância**
   (sessão em memória) e serializa no JSON da sessão. Os nós validam a entrada
   aceitando modelo ou dict.
3. **MCP: `SessionMcpGateway` é o padrão**, por três motivos:
   a. invocar uma tool do `McpToolset` fora de um agente exige um
      `tool_context` obrigatório (funciona com `None`, sem garantia);
   b. a criação de sessão tem um retry interno (`retry_on_errors`), que
      somaria um segundo nível de retry ao `RetryConfig` do nó;
   c. falhas de transporte viram `ConnectionError` com texto, sem distinguir
      um 401 (permanente) de uma queda de rede (transitória).
   O gateway usa `mcp.client.sse.sse_client` + `ClientSession.call_tool`, com
   os timeouts da spec, uma conexão por chamada. `AdkMcpGateway` existe como
   alternativa (`AGENT_MCP_GATEWAY=adk`), coberto pelo teste de integração.
4. **`before_model_callback` que devolve `LlmResponse` não falha o nó**: o
   modelo não é chamado, o `Agent` termina sem saída e o fluxo segue. O
   `pii_tripwire` devolve a resposta de erro e marca `run.pii_tripwire` no
   estado; o `reconcile` falha com `PermanentError("PII_TRIPWIRE")` antes de
   qualquer resolução ou agendamento.
5. **Exceção em nó encerra a execução**: `Runner.run_async` relança a exceção
   original e a sessão guarda um evento com `error_code` = nome da classe e
   `error_message` = `str(exc)`, e o ADK registra o traceback no logger
   `google_adk`. Mensagens de exceção do runtime nunca levam texto do
   documento, e a CLI aplica o filtro de logs de biblioteca ao `google_adk`.
6. **Retry**: as factories copiam o `RetryConfig` do código gerado com
   `exceptions=[TransientError, NodeTimeoutError]`. `PermanentError` não é
   re-tentada.
7. **Resume reemite eventos**: ao retomar uma invocação interrompida, o
   `Runner` devolve de novo os eventos dos nós já concluídos como eventos
   novos (id e timestamp novos, sem rota). A CLI ignora eventos já vistos em
   uma rodada anterior, por id ou por conteúdo (caminho do nó, erro e saída),
   para não registrar nós duas vezes.

## Consequências

- O formato do código gerado e os golden files continuam os da fase 02.
- Um upgrade do ADK começa por este teste; cada falha aponta a suposição e o
  item acima diz o que mudar.
- O gateway MCP não usa classes do ADK. Se uma versão futura expuser uma
  invocação direta sem contexto e sem retry interno, o `AdkMcpGateway` passa
  a ser o padrão sem mudar spec nem código gerado.
- Uma conexão SSE por chamada custa alguns milissegundos e evita sessões
  abertas entre tarefas diferentes do ADK (cancel scopes do anyio).
