# Execuções coletadas

Gerado por `scripts/collect_evidence.py`; não editar à mão.

Cada cenário roda uma vez. A captura em tabela (`cli/`) e, quando há, o relatório JSON (`runs/`) são da mesma execução: o `run_id` do relatório é o da última linha da captura.

| Cenário | LLM | Exit | Outcome | Agendamento | Captura | Relatório |
|---|---|---|---|---|---|---|
| Fluxo feliz | gemini | 0 | scheduled | solicitação registrada | [`cli/01-feliz.txt`](../cli/01-feliz.txt) | [`runs/d019f4ce-c6a4-4a16-9e10-087be2836110.json`](./d019f4ce-c6a4-4a16-9e10-087be2836110.json) |
| Reexecução (replay idempotente) | gemini | 0 | replayed | solicitação já existente (replay idempotente) | [`cli/02-replay.txt`](../cli/02-replay.txt) | [`runs/b3a1cbb1-3ea0-45c5-8633-9e1e486ff15b.json`](./b3a1cbb1-3ea0-45c5-8633-9e1e486ff15b.json) |
| Injeção de prompt | sem LLM | 3 | INPUT_BLOCKED | — | [`cli/03-injection.txt`](../cli/03-injection.txt) | [`runs/9b70ac58-dfd6-4272-91d9-9ac8d0fe04f0.json`](./9b70ac58-dfd6-4272-91d9-9ac8d0fe04f0.json) |
| Pedido ambíguo, não interativo | gemini | 0 | scheduled | solicitação registrada | [`cli/04-ambiguo.txt`](../cli/04-ambiguo.txt) | [`runs/54cd7cd7-b3c3-4286-87b2-dab51b527e7b.json`](./54cd7cd7-b3c3-4286-87b2-dab51b527e7b.json) |
| Imagem ilegível | sem LLM | 4 | OCR_LOW_QUALITY | — | [`cli/05-ilegivel.txt`](../cli/05-ilegivel.txt) | — |
| Spec alternativa (exam_lookup_only) | sem LLM | 0 | resolved | exames identificados (sem agendamento) | [`cli/06-spec-alternativa.txt`](../cli/06-spec-alternativa.txt) | — |
| Spec alterada sem regenerar | — | 7 | — | — | [`cli/07-spec-alterada.txt`](../cli/07-spec-alterada.txt) | — |
