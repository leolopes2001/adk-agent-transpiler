# Execuções coletadas

Gerado por `scripts/collect_evidence.py`; não editar à mão.

Primeiro as execuções com `--output json` (relatórios em `runs/`); depois o banco é esvaziado (`TRUNCATE appointments CASCADE`) e os mesmos cenários rodam com a saída em tabela (`cli/`), para que as duas séries comecem com um agendamento novo.

| Cenário | Saída | LLM | Exit | Outcome | Agendamento | Arquivo |
|---|---|---|---|---|---|---|
| Fluxo feliz | `--output json` | gemini | 0 | scheduled | scheduled | [`runs/9c2be3f2-fadc-4637-bd95-d49ba7f42fba.json`](./9c2be3f2-fadc-4637-bd95-d49ba7f42fba.json) |
| Reexecução (replay idempotente) | `--output json` | gemini | 0 | replayed | replayed | [`runs/620d772a-52e4-4f64-b0e8-64685eb93657.json`](./620d772a-52e4-4f64-b0e8-64685eb93657.json) |
| Injeção de prompt | `--output json` | sem LLM | 3 | INPUT_BLOCKED | — | [`runs/5a34871e-1456-4566-854b-e6e19f34eb56.json`](./5a34871e-1456-4566-854b-e6e19f34eb56.json) |
| Pedido ambíguo, não interativo | `--output json` | gemini | 0 | scheduled | scheduled | [`runs/463df54a-e242-4ca2-b22c-73219742153a.json`](./463df54a-e242-4ca2-b22c-73219742153a.json) |
| Fluxo feliz | `--output table` | gemini | 0 | scheduled | solicitação registrada | [`cli/01-feliz.txt`](../cli/01-feliz.txt) |
| Reexecução (replay idempotente) | `--output table` | gemini | 0 | replayed | solicitação já existente (replay idempotente) | [`cli/02-replay.txt`](../cli/02-replay.txt) |
| Injeção de prompt | `--output table` | sem LLM | 3 | INPUT_BLOCKED | — | [`cli/03-injection.txt`](../cli/03-injection.txt) |
| Pedido ambíguo, não interativo | `--output table` | gemini | 0 | scheduled | solicitação registrada | [`cli/04-ambiguo.txt`](../cli/04-ambiguo.txt) |
| Imagem ilegível | `--output table` | sem LLM | 4 | OCR_LOW_QUALITY | — | [`cli/05-ilegivel.txt`](../cli/05-ilegivel.txt) |
| Spec alternativa (exam_lookup_only) | `padrão da spec` | sem LLM | 0 | resolved | exames identificados (sem agendamento) | [`cli/06-spec-alternativa.txt`](../cli/06-spec-alternativa.txt) |
| Spec alterada sem regenerar | — | — | 7 | — | — | [`cli/07-spec-alterada.txt`](../cli/07-spec-alterada.txt) |
