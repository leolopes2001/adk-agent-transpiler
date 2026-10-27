# ADR-0001 — Orquestrar o agente como Workflow em grafo do ADK 2

- **Status:** Aceito
- **Data:** 2026-10-27

## Contexto

O agente precisa ler um pedido médico, identificar exames, resolver códigos e
registrar um agendamento. A forma mais direta com ADK seria um único agente
LLM com ferramentas, decidindo sozinho a ordem das chamadas. Nesse arranjo, o
modelo decide se e o que agendar, retries dependem do prompt e controles de
PII e de prompt injection dependem de o modelo cooperar. O texto de entrada
vem de OCR, ou seja, de um documento que não controlamos.

## Decisão

O fluxo é um `Workflow` em grafo do ADK 2. Cada etapa é um nó com entrada e
saída tipadas, e as transições são rotas explícitas (`ok`, `low_quality`,
`blocked`, `accepted`, `review`, `empty`, ...) emitidas por código.

O LLM aparece em um único ponto: um `Agent` single-turn, sem ferramentas, que
transforma texto já mascarado em uma lista estruturada de menções de exames.
Códigos, aceitação, roteamento, retries, timeouts e o agendamento são
decididos por nós determinísticos.

## Consequências

- O caminho até o agendamento é verificável: o transpilador pode provar que
  toda rota até `schedule` passa pelo gate de confiança.
- Cada nó é uma função testável isoladamente; o LLM pode ser substituído por
  um fake em testes sem mudar o grafo.
- Retries e timeouts são configuração por nó, não instrução em prompt.
- Custo: o grafo precisa ser descrito na spec e validado, o que exige um
  transpilador mais elaborado do que gerar um agente único.
- A API do Workflow do ADK 2 é recente; as suposições sobre ela estão fixadas
  em testes de contrato (`tests/contract/test_adk_contract.py`, ADR-0009).
