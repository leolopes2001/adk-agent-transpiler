# ADR-0005 — Busca léxica híbrida no catálogo, com thresholds calibrados

- **Status:** Aceito
- **Data:** 2026-10-27

## Contexto

O catálogo tem 120 exames fictícios (`data/exams_catalog.json`). As consultas
são menções curtas extraídas de um pedido: abreviações ("HMG", "TGO"), erros
de OCR ("Hemogrma") e variações de escrita. O resultado alimenta uma decisão
automática de agendar ou não. Embeddings genéricos exigiriam baixar modelos
grandes e não lidam bem com siglas técnicas nem com erros de caractere.

## Decisão

A busca combina BM25 sobre tokens normalizados com um índice de trigramas de
caracteres e reordena os candidatos por similaridade fuzzy, produzindo um
score em [0, 1]. Match exato de nome ou alias vale 1.0. A normalização é
`lab_common.text.normalize_for_search`, a mesma usada no runtime para checar
se o que o LLM extraiu aparece no texto.

O servidor só recupera e pontua. Quem decide é o nó `confidence_gate`, com
thresholds definidos na spec (0,85 / 0,60 / margem 0,05) e calibrados por
`scripts/calibrate_rag.py` contra um conjunto de avaliação versionado
(`tests/data/rag_eval.jsonl`). A regra do gate fica em
`lab_common.gate.gate_decision`, usada pelo nó, pelo teste de regressão do
RAG e pelo script de calibração; a margem é comparada com 4 casas
(`round(c1 - c2, 4)`), a mesma precisão dos scores.

## Consequências

- Sem dependências pesadas nem chamadas externas; os scores são
  determinísticos e explicáveis.
- O catálogo precisa de aliases bem cuidados; um teste garante que nomes e
  aliases normalizados são únicos e que termos ambíguos ("T4", "colesterol")
  continuam sem match exato.
- Os thresholds são configuração, não código; mudar a política não exige
  mexer no servidor.
- Embeddings ficam como retriever opcional, a ser ligado apenas se mostrarem
  ganho mensurável sobre o conjunto de avaliação.
