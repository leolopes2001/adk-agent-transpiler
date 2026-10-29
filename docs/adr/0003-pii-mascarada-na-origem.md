# ADR-0003 — Mascarar PII na origem, com verificações adicionais depois

- **Status:** Aceito
- **Data:** 2026-10-27

## Contexto

A imagem do pedido traz nome, CPF, telefone, data de nascimento e outros
dados pessoais. O desafio exige que PII não chegue ao LLM nem seja
persistida. O texto passa por várias fronteiras depois do OCR: eventos de
sessão do ADK, chamada ao modelo, logs, API de agendamento.

## Decisão

O servidor OCR detecta e mascara PII antes de devolver qualquer texto. Se o
motor de PII falhar, a tool falha sem texto (fail-closed). O texto bruto
nunca sai desse processo.

Depois disso, outras camadas verificam de forma independente:

1. o nó `guard` do runtime reescaneia o texto recebido e mascara de novo se
   encontrar algo;
2. um `before_model_callback` (`agent_runtime.guardrails.tripwire`) escaneia
   a requisição inteira ao LLM e bloqueia a chamada se houver identificador
   válido ou parte que não consegue inspecionar; o `reconcile` então falha a
   execução com `PII_TRIPWIRE` (ADR-0009, item 4);
3. um processador de logs (`RedactPii`, em `lab_common.logging`, ligado por
   padrão em todos os serviços) remove chaves proibidas e redige valores, e
   um filtro corta o payload dos logs das bibliotecas `mcp`, `uvicorn` e
   `google_adk`;
4. a API recebe só um `patient_ref` opaco (HMAC do identificador), códigos e
   nomes de exames; o contrato não tem campo para dados pessoais.

### NER local e minimização (2026-10-29)

As regras só achavam nomes depois de rótulos ("Paciente:", "Dr."): 8 de 27 no
corpus sintético (`tests/data/pii_ner_cases.jsonl`). Somamos um detector de
nomes com Presidio e spaCy `pt_core_news_md` (`pii_guard.ner`), rodando no
OCR, no `guard` e no tripwire, sobre o texto e cada linha, no original e numa
cópia capitalizada: 25 de 27 (os 2 restantes são sobrenomes iguais a termos
médicos, documentados como escape), sem mascarar nenhum termo médico do corpus nem
nenhum dos 3.100 testes do catálogo. Seus spans perdem toda sobreposição para
as regras e são cortados em termos do catálogo, termos médicos e rótulos.
LOCATION ficou de fora (pouco ganho, falsos positivos em exames). O nó
`llm_extract` passou a enviar ao modelo só linhas candidatas a exame; linhas
com rótulo de dado pessoal nunca vão, nem mascaradas. Custo: imagem do OCR de
681 MB para 1,11 GB e do agente de 382 MB para 878 MB; cerca de 1,4 s para
carregar o modelo na subida; mascarar um pedido passa de ~1 ms para 80–125 ms.
O que continua escapando está no README, seção "Segurança".

## Consequências

- Um defeito em uma camada não basta para vazar PII.
- O detector precisa de alta precisão para identificadores com dígito
  verificador (CPF, CNS) e de cuidado para não mascarar nomes de exames;
  isso é coberto por teste cruzado com o catálogo.
- O operador nunca vê o nome do paciente na saída; a correlação é feita pelo
  `patient_ref`.
- O segredo do HMAC passa a ser crítico e fica restrito ao servidor OCR.
- Implementação: `pii_guard` (detectores, NER em `pii_guard.ner` e redator),
  `ocr_mcp.server` e `ocr_mcp.identity` (mascaramento e `patient_ref`),
  `lab_common.logging`, `agent_runtime.nodes.untrusted_text_guard`,
  `agent_runtime.nodes.llm_input` (minimização) e
  `agent_runtime.guardrails.tripwire`.
