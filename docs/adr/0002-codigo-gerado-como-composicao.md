# ADR-0002 — Código gerado só compõe factories de uma biblioteca de runtime

- **Status:** Aceito
- **Data:** 2026-10-27

## Contexto

O transpilador recebe uma spec JSON e precisa emitir Python executável com
Google ADK. Gerar a lógica de cada nó dentro do template tornaria o código
gerado longo, difícil de testar (cada spec geraria uma variação) e abriria
espaço para injeção de código a partir de campos da spec.

## Decisão

O código gerado contém apenas a fiação do grafo: imports, chamadas a
factories de `agent_runtime.nodes` com parâmetros literais vindos da spec,
arestas e rotas. Toda a lógica de negócio vive em `agent_runtime`, que é uma
biblioteca comum com testes próprios.

Para validar a ideia cedo, a fase 02 entrega um *walking skeleton*: factories
com a assinatura final que ainda levantam `NotImplementedError`. O código
gerado já importa e monta o grafo; a fase 06 preenche as implementações sem
mudar o que o transpilador emite.

## Consequências

- O código gerado é curto e revisável; golden files mostram exatamente o que
  muda entre specs.
- A lógica é testada uma vez, na biblioteca, e não por spec.
- A superfície de injeção diminui: o template só insere identificadores
  validados por regex e literais via `repr`.
- Assinaturas das factories viram contrato entre transpilador e runtime; um
  teste de contrato compara o registry de node kinds com as factories.
- O pacote gerado depende de `agent_runtime` instalado na mesma versão; isso
  é garantido pela imagem Docker do agente.
