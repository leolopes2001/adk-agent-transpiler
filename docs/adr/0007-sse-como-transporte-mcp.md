# ADR-0007 — SSE como transporte MCP, isolado atrás do gateway

- **Status:** Aceito
- **Data:** 2026-10-27

## Contexto

O desafio exige que OCR e RAG sejam servidores MCP acessados via SSE. No
protocolo MCP atual, SSE é o transporte legado; o recomendado é Streamable
HTTP. O SDK oficial `mcp` também mudou: na série 2.x, a classe `FastMCP` foi
renomeada para `MCPServer` (módulo `mcp.server.mcpserver`) e outras APIs
mudaram. O `google-adk` 2.13.0 aceita `mcp>=1.24,<3`.

Verificamos com `mcp` 2.3.0: o cliente `mcp.client.sse.sse_client` e o
`McpToolset` com `SseConnectionParams` do ADK funcionam, mas
`mcp.server.fastmcp.FastMCP` não existe mais.

## Decisão

- Os servidores usam `FastMCP` com `sse_app()`; o runtime conecta com
  `McpToolset` + `SseConnectionParams`.
- O `pyproject.toml` fixa `mcp>=1.24,<2`. O limite inferior é o mínimo aceito
  pelo `google-adk`; o superior evita a quebra de API da série 2.x. O lock
  resolveu para `mcp` 1.30.0.
- O transporte fica restrito a dois pontos: a montagem do app no servidor e
  os parâmetros de conexão no gateway do runtime.

## Consequências

- Atende ao requisito sem espalhar detalhes de SSE pelo código.
- Ficamos na série 1.x do SDK; correções que só existirem na 2.x não chegam
  automaticamente.
- Evolução futura: migrar para `mcp` 2.x (`MCPServer`) e para Streamable HTTP.
  Pelo isolamento acima, a troca deve afetar só os servidores e o gateway.
- Conexões SSE são longas; headers são fixados na conexão, então a
  correlação por chamada vai como argumento das tools.
