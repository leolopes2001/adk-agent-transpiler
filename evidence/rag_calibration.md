# Calibração dos thresholds do confidence gate

Gerado por `scripts/calibrate_rag.py`; não editar à mão.

## Entradas

- Conjunto de avaliação: `tests/data/rag_eval.jsonl` (103 consultas; sha256 `706772e20447bb3a…`).
- Catálogo: `data/exams_catalog.json`, versão 2026.10.0 (sha256 `28eb7825865727be…`).
- Thresholds atuais: `specs/exam_scheduling_agent.json`.
- Consultas por tipo: abbrev 15, alias 12, ambiguous 8, exact 17, negative 26, ocr_noise 13, typo 12.

## Método

Cada consulta é buscada uma vez (top 5) com o catálogo e o scorer reais. A regra do gate é replicada: aceita se `c1 ≥ accept` e (`c2` ausente ou `c1 - c2 ≥ margin`); revisão se `c1 ≥ review`; senão rejeita.

Grade: accept 0.70 a 0.95 (passo 0.05), review 0.45 a 0.80 (passo 0.05, review ≤ accept), margin 0.00 a 0.15 (passo 0.01).

- **Precisão dos aceitos**: aceitos com o código certo / aceitos (ambíguo aceito conta como erro).
- **Recall dos aceitos**: aceitos certos / consultas com uma única resposta certa (exceto negativas e ambíguas).
- **Taxa de revisão**: consultas enviadas para revisão / todas as consultas.
- **Positivos rejeitados**: consultas com resposta certa que o gate descartaria sem revisão.

Critério: precisão dos aceitos = 100% e zero negativos aceitos; entre essas combinações, maior recall; empate → a mais próxima dos thresholds atuais.

## Resultado

| | accept / review / margin | precisão dos aceitos | recall dos aceitos | taxa de revisão | negativos aceitos | ambíguos aceitos | positivos rejeitados |
|---|---|---|---|---|---|---|---|
| atual (spec) | 0.85 / 0.60 / 0.05 | 100.0% | 79.7% (55/69) | 31.1% (32/103) | 0 | 0 | 1 |
| **escolhido** | 0.85 / 0.60 / 0.05 | 100.0% | 79.7% (55/69) | 31.1% (32/103) | 0 | 0 | 1 |
| grade fina (0,01), só referência | 0.81 / 0.60 / 0.05 | 100.0% | 87.0% (60/69) | 26.2% (27/103) | 0 | 0 | 1 |

Precision@1 nas consultas não negativas: 98.7% (76/77).

Os thresholds escolhidos são os que já estão na spec; nada muda.

Combinações admissíveis: 360; com o mesmo recall máximo: 56.

## Decisões por consulta (thresholds escolhidos)

| tipo | consulta | esperado | top-1 | score | margem | decisão |
|---|---|---|---|---|---|---|
| exact | `1. Hemograma completo` | LAB-10001 | LAB-10001 | 1.0000 | 0.5012 | accepted |
| exact | `2. Glicemia de jejum` | LAB-10012 | LAB-10012 | 1.0000 | 0.2338 | accepted |
| exact | `3. TSH` | LAB-10043 | LAB-10043 | 1.0000 | 0.5000 | accepted |
| exact | `4. T4 livre` | LAB-10044 | LAB-10044 | 1.0000 | 0.2188 | accepted |
| exact | `5. Colesterol total e frações` | LAB-10016 | LAB-10016 | 1.0000 | 0.2754 | accepted |
| exact | `8. Triglicerídeos` | LAB-10020 | LAB-10020 | 1.0000 | 0.4937 | accepted |
| exact | `Creatinina` | LAB-10021 | LAB-10021 | 1.0000 | 0.0750 | accepted |
| exact | `Ureia` | LAB-10022 | LAB-10022 | 1.0000 | 0.4231 | accepted |
| exact | `Ferritina` | LAB-10109 | LAB-10109 | 1.0000 | 0.4000 | accepted |
| exact | `Urocultura` | LAB-10099 | LAB-10099 | 1.0000 | 0.3864 | accepted |
| exact | `Hemoglobina glicada` | LAB-10013 | LAB-10013 | 1.0000 | 0.3525 | accepted |
| exact | `PSA total` | LAB-10093 | LAB-10093 | 1.0000 | 0.2338 | accepted |
| exact | `Beta-HCG` | LAB-10049 | LAB-10049 | 1.0000 | 0.5500 | accepted |
| exact | `Ultrassonografia de abdome total` | LAB-10115 | LAB-10115 | 1.0000 | 0.1921 | accepted |
| exact | `ÁCIDO ÚRICO` | LAB-10030 | LAB-10030 | 1.0000 | 0.2880 | accepted |
| exact | `Vitamina B12` | LAB-10108 | LAB-10108 | 1.0000 | 0.2417 | accepted |
| exact | `Tempo de protrombina` | LAB-10087 | LAB-10087 | 1.0000 | 0.1976 | accepted |
| alias | `6. Urina tipo 1 (EAS)` | LAB-10079 | LAB-10079 | 0.9000 | 0.3600 | accepted |
| alias | `glicose` | LAB-10012 | LAB-10012 | 1.0000 | 0.0750 | accepted |
| alias | `lipidograma` | LAB-10016 | LAB-10016 | 1.0000 | 0.4600 | accepted |
| alias | `perfil lipídico` | LAB-10016 | LAB-10016 | 1.0000 | 0.4937 | accepted |
| alias | `urina rotina` | LAB-10079 | LAB-10079 | 1.0000 | 0.2338 | accepted |
| alias | `RT-PCR covid` | LAB-10104 | LAB-10104 | 1.0000 | 0.2417 | accepted |
| alias | `Vitamina D` | LAB-10107 | LAB-10107 | 1.0000 | 0.2000 | accepted |
| alias | `tiroxina livre` | LAB-10044 | LAB-10044 | 1.0000 | 0.2338 | accepted |
| alias | `3) TGP` | LAB-10024 | LAB-10024 | 1.0000 | 0.3250 | accepted |
| alias | `- Parasitológico de fezes (EPF)` | LAB-10080 | LAB-10080 | 0.8792 | 0.1546 | accepted |
| alias | `Colesterol HDL` | LAB-10017 | LAB-10017 | 1.0000 | 0.1786 | accepted |
| alias | `Urocultura com antibiograma` | LAB-10099 | LAB-10099 | 1.0000 | 0.4477 | accepted |
| abbrev | `7. PCR` | LAB-10037 | LAB-10037 | 1.0000 | 0.0750 | accepted |
| abbrev | `PCR` | LAB-10037 | LAB-10037 | 1.0000 | 0.0750 | accepted |
| abbrev | `HMG` | LAB-10001 | LAB-10001 | 1.0000 | 0.3250 | accepted |
| abbrev | `HbA1c` | LAB-10013 | LAB-10013 | 1.0000 | 0.4937 | accepted |
| abbrev | `TGO` | LAB-10023 | LAB-10023 | 1.0000 | 0.3250 | accepted |
| abbrev | `T4L` | LAB-10044 | LAB-10044 | 1.0000 | 0.4600 | accepted |
| abbrev | `EAS` | LAB-10079 | LAB-10079 | 1.0000 | 0.4600 | accepted |
| abbrev | `EPF` | LAB-10080 | LAB-10080 | 1.0000 | 0.5500 | accepted |
| abbrev | `TP/INR` | LAB-10087 | LAB-10087 | 1.0000 | 0.4600 | accepted |
| abbrev | `VHS` | LAB-10004 | LAB-10004 | 1.0000 | 0.6625 | accepted |
| abbrev | `GGT` | LAB-10025 | LAB-10025 | 1.0000 | 0.5500 | accepted |
| abbrev | `TTPA` | LAB-10088 | LAB-10088 | 1.0000 | 0.3250 | accepted |
| abbrev | `BHCG` | LAB-10049 | LAB-10049 | 1.0000 | 0.6143 | accepted |
| abbrev | `USG abdome total` | LAB-10115 | LAB-10115 | 1.0000 | 0.1921 | accepted |
| abbrev | `glic jejum` | LAB-10012 | LAB-10012 | 1.0000 | 0.3750 | accepted |
| typo | `Hemogrma completo` | LAB-10001 | LAB-10001 | 0.8536 | 0.3627 | accepted |
| typo | `Glicemla de jejum` | LAB-10012 | LAB-10012 | 0.8214 | 0.2143 | review |
| typo | `Creatinlna` | LAB-10021 | LAB-10021 | 0.6750 | 0.0000 | review |
| typo | `Urocultra` | LAB-10099 | LAB-10099 | 0.7105 | 0.1391 | review |
| typo | `Ferritlna` | LAB-10109 | LAB-10031 ✗ | 0.6750 | 0.0083 | review |
| typo | `Colesterol totl e frações` | LAB-10016 | LAB-10016 | 0.9007 | 0.1761 | accepted |
| typo | `Hemoglobina glicda` | LAB-10013 | LAB-10013 | 0.8547 | 0.1892 | accepted |
| typo | `Potásio` | LAB-10032 | LAB-10032 | 0.7000 | 0.2500 | review |
| typo | `Fosfatase alcalna` | LAB-10026 | LAB-10026 | 0.8536 | 0.2305 | accepted |
| typo | `Albumna` | LAB-10029 | LAB-10029 | 0.7000 | 0.0250 | review |
| typo | `Testosterona totl` | LAB-10055 | LAB-10055 | 0.8536 | 0.3344 | accepted |
| typo | `Vitamna B12` | LAB-10108 | LAB-10108 | 0.8424 | 0.1674 | review |
| ocr_noise | `1. Hemograrna completo` | LAB-10001 | LAB-10001 | 0.8142 | 0.3233 | review |
| ocr_noise | `2. GIicemia de jejum` | LAB-10012 | LAB-10012 | 0.8214 | 0.2143 | review |
| ocr_noise | `3. T5H` | LAB-10043 | LAB-10043 | 0.5000 | 0.0500 | rejected |
| ocr_noise | `4. T4 Iivre` | LAB-10044 | LAB-10044 | 0.7812 | 0.1162 | review |
| ocr_noise | `5. Colesterol total e fraçöes` | LAB-10016 | LAB-10016 | 1.0000 | 0.2754 | accepted |
| ocr_noise | `6. Urina tipo l (EAS)` | LAB-10079 | LAB-10079 | 0.8000 | 0.0963 | review |
| ocr_noise | `7. PCR.` | LAB-10037 | LAB-10037 | 1.0000 | 0.0750 | accepted |
| ocr_noise | `Hemograma cornpleto` | LAB-10001 | LAB-10001 | 0.8142 | 0.3154 | review |
| ocr_noise | `Proteina C reatlva` | LAB-10037 | LAB-10037 | 0.8750 | 0.1504 | accepted |
| ocr_noise | `Vitamlna D (25-0H)` | LAB-10107 | LAB-10107 | 0.7812 | 0.0775 | review |
| ocr_noise | `\| Ferritina` | LAB-10109 | LAB-10109 | 1.0000 | 0.4000 | accepted |
| ocr_noise | `Creatinina ;` | LAB-10021 | LAB-10021 | 1.0000 | 0.0750 | accepted |
| ocr_noise | `Glicemia de jejum —` | LAB-10012 | LAB-10012 | 1.0000 | 0.2338 | accepted |
| ambiguous | `T4` | LAB-10044/LAB-10045 | LAB-10044 | 0.9250 | 0.0000 | review |
| ambiguous | `4. T4` | LAB-10044/LAB-10045 | LAB-10044 | 0.9250 | 0.0000 | review |
| ambiguous | `T3` | LAB-10046/LAB-10047 | LAB-10046 | 0.9250 | 0.0000 | review |
| ambiguous | `colesterol` | LAB-10016/LAB-10017/LAB-10018/LAB-10019 | LAB-10017 | 0.9625 | 0.0000 | review |
| ambiguous | `5. Colesterol` | LAB-10016/LAB-10017/LAB-10018/LAB-10019 | LAB-10017 | 0.9625 | 0.0000 | review |
| ambiguous | `Ultrassonografia` | LAB-10115/LAB-10116/LAB-10117/LAB-10118 | LAB-10115 | 0.9250 | 0.0000 | review |
| ambiguous | `Hemoglobina` | LAB-10001/LAB-10010/LAB-10013 | LAB-10010 | 0.9250 | 0.0000 | review |
| ambiguous | `Cultura` | LAB-10099/LAB-10100/LAB-10101/LAB-10102 | LAB-10099 | 0.9250 | 0.0000 | review |
| negative | `Ressonância magnética de crânio` | — | LAB-10034 | 0.5063 | 0.0788 | rejected |
| negative | `Tomografia de tórax` | — | LAB-10119 | 0.7662 | 0.2797 | review |
| negative | `Eletrocardiograma` | — | LAB-10001 | 0.5400 | 0.0000 | rejected |
| negative | `Colonoscopia` | — | LAB-10103 | 0.5063 | 0.0563 | rejected |
| negative | `Mamografia bilateral` | — | LAB-10075 | 0.5400 | 0.0535 | rejected |
| negative | `Ecocardiograma` | — | LAB-10001 | 0.5400 | 0.0000 | rejected |
| negative | `Endoscopia digestiva alta` | — | LAB-10011 | 0.4909 | 0.0409 | rejected |
| negative | `Vitamina E` | — | LAB-10107 | 0.9625 | 0.0000 | review |
| negative | `Testosterona livre` | — | LAB-10055 | 0.8000 | 0.0338 | review |
| negative | `Cálcio iônico` | — | LAB-10033 | 0.8000 | 0.2600 | review |
| negative | `Ultrassonografia de mamas` | — | LAB-10115 | 0.7250 | 0.0000 | review |
| negative | `Holter 24 horas` | — | LAB-10085 | 0.7238 | 0.1838 | review |
| negative | `Cortisol salivar` | — | LAB-10056 | 0.8000 | 0.3091 | review |
| negative | `[PESSOA_1]` | — | LAB-10061 | 0.7662 | 0.0000 | review |
| negative | `Paciente: [PESSOA_1]` | — | LAB-10009 | 0.5400 | 0.0000 | rejected |
| negative | `CPF: [CPF_1]` | — | LAB-10041 | 0.5400 | 0.0270 | rejected |
| negative | `Dr. [PESSOA_2] — [CRM_1]` | — | LAB-10105 | 0.6913 | 0.1913 | review |
| negative | `Belo Horizonte, 20/10/2026` | — | LAB-10049 | 0.3857 | 0.0482 | rejected |
| negative | `Indicação clínica: avaliação de rotina.` | — | LAB-10079 | 0.7037 | 0.1974 | review |
| negative | `SOLICITAÇÃO DE EXAMES LABORATORIAIS` | — | LAB-10030 | 0.5400 | 0.0900 | rejected |
| negative | `Laboratório Exemplo (fictício)` | — | LAB-10030 | 0.5400 | 0.0675 | rejected |
| negative | `Convênio: Plano Exemplo Saúde (fictício)` | — | LAB-10033 | 0.4500 | 0.0643 | rejected |
| negative | `Exames solicitados:` | — | LAB-10008 | 0.6750 | 0.2250 | review |
| negative | `\|\|\|\| ~~ ;;` | — | — | 0.0000 | — | rejected |
| negative | `lI1l \|` | — | LAB-10016 | 0.4500 | 0.0000 | rejected |
| negative | `xkqzv wpt` | — | — | 0.0000 | — | rejected |

## Folgas

Negativos com maior score (top-1):

- `Vitamina E` → LAB-10107 0.9625 (margem 0.0000; review)
- `Testosterona livre` → LAB-10055 0.8000 (margem 0.0338; review)
- `Cálcio iônico` → LAB-10033 0.8000 (margem 0.2600; review)
- `Cortisol salivar` → LAB-10056 0.8000 (margem 0.3091; review)
- `Tomografia de tórax` → LAB-10119 0.7662 (margem 0.2797; review)

Aceitos com menor score:

- `Hemogrma completo` → LAB-10001 0.8536
- `Fosfatase alcalna` → LAB-10026 0.8536
- `Testosterona totl` → LAB-10055 0.8536
- `Hemoglobina glicda` → LAB-10013 0.8547
- `Proteina C reatlva` → LAB-10037 0.8750

## Limitações

- O conjunto é pequeno e foi escrito à mão; os thresholds valem para ele. Pedidos reais devem alimentar o conjunto antes de qualquer ajuste fino.
- O passo de 0,05 em accept/review é proposital: com passo 0,01 o accept cairia no intervalo estreito entre o negativo de maior score sem empate e o positivo mais fraco.
