# ADR-0004 — Chamar os servidores MCP a partir de nós determinísticos

- **Status:** Aceito
- **Data:** 2026-10-27

## Contexto

OCR e busca no catálogo são expostos como servidores MCP. O uso mais comum de
MCP com ADK é entregar o `McpToolset` a um agente LLM e deixar o modelo
decidir quando chamar cada ferramenta. Aqui o modelo processa texto vindo de
um documento externo, que pode conter instruções maliciosas. Dar ferramentas
a esse modelo permitiria que o documento influenciasse quais chamadas são
feitas.

## Decisão

As ferramentas MCP são chamadas por nós determinísticos do workflow (`ocr`,
`lexical_extract`, `resolve`) por meio de um gateway no runtime
(`agent_runtime.gateways.mcp`). O gateway aplica a allowlist de ferramentas
declarada na spec e traduz erros de transporte para erros do domínio.

A implementação padrão é o `SessionMcpGateway`, sobre o cliente SSE do SDK
`mcp`; o `AdkMcpGateway`, sobre o `McpToolset` do ADK, fica como alternativa.
Os motivos estão no ADR-0009, item 3.

O agente LLM não recebe nenhuma ferramenta. O transpilador rejeita specs em
que um agente que lê texto não confiável tenha ferramentas.

## Consequências

- Uma instrução injetada no pedido não consegue disparar chamadas MCP nem
  agendamentos.
- Os argumentos de cada chamada são montados por código e podem ser
  validados e testados.
- Perde-se a flexibilidade de o modelo decidir buscar mais informação; para
  este caso de uso o fluxo é fixo, então isso não faz falta.
- O gateway é uma porta (`McpGateway`) com implementação fake para testes.
- A allowlist é verificada duas vezes: pelo transpilador
  (`E_TOOL_NOT_ALLOWED`) e pelo gateway, em tempo de execução.
