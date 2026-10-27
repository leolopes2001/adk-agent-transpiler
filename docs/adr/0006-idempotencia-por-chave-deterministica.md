# ADR-0006 — Idempotência por chave determinística e unique constraint

- **Status:** Aceito
- **Data:** 2026-10-27

## Contexto

O agendamento é o único efeito colateral do fluxo. A chamada HTTP à API pode
ser repetida por retry automático, por timeout com gravação já feita, ou
porque o operador rodou a CLI de novo com a mesma imagem. Nenhum desses casos
deve criar duas solicitações.

## Decisão

O agente calcula a `Idempotency-Key` a partir do conteúdo:
`sha256("v1|<hash do documento>|<patient_ref>|<códigos aceitos ordenados>")`.
Reexecutar com a mesma imagem e o mesmo resultado gera a mesma chave.

A API guarda a chave com o hash canônico do corpo da requisição, em uma
tabela cuja chave primária é a própria `Idempotency-Key`:

- chave nova: cria a solicitação (201);
- mesma chave e mesmo corpo: devolve a resposta original (200, com header
  indicando replay);
- mesma chave e corpo diferente: 409.

Duas requisições simultâneas com a mesma chave são resolvidas pelo Postgres:
a segunda falha na constraint e relê o registro gravado pela primeira.

## Consequências

- Retries são seguros; o nó de agendamento pode ter `RetryConfig` sem risco
  de duplicar.
- A garantia vale sob concorrência porque está no banco, não em memória.
- A chave expira (`IDEMPOTENCY_TTL_HOURS`, padrão 24 h), então uma
  reexecução muito posterior cria uma nova solicitação; isso é aceitável para
  o caso de uso.
- A validação do catálogo vem antes da idempotência: uma requisição inválida
  não grava nem consome a chave.
- Implementação: `scheduler_api.domain.services` e
  `scheduler_api.infra.repositories` (API), `agent_runtime.nodes.schedule`
  (chave e cliente).
