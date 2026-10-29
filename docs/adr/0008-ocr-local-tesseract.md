# ADR-0008 — OCR local com Tesseract em vez de LLM multimodal

- **Status:** Aceito
- **Data:** 2026-10-27

## Contexto

O primeiro passo do fluxo é extrair texto da imagem de um pedido médico. Um
LLM multimodal faria isso com boa qualidade, inclusive para manuscritos, mas
exigiria enviar a imagem inteira, com todos os dados pessoais, a um serviço
externo. Isso contradiz o requisito de PII e a decisão de mascarar na origem
(ADR-0003).

## Decisão

O servidor OCR usa Tesseract com o modelo de português, rodando localmente no
container, com pré-processamento em OpenCV/Pillow (escala de cinza,
binarização, correção de inclinação). A imagem nunca sai do servidor OCR; só
texto mascarado é devolvido.

O binário do Tesseract é chamado por subprocesso, sem shell, com a imagem no
stdin e a tabela TSV no stdout. O `pytesseract` não é usado porque grava a
imagem e o texto reconhecido em arquivos temporários; assim, nada do
documento toca o disco.

As imagens de teste são fictícias e versionadas em `samples/` e
`tests/data/images/`, para que a qualidade do OCR possa ser medida sobre
entradas fixas.

## Consequências

- Nenhum dado pessoal é enviado a terceiros na etapa de OCR.
- Resultado determinístico e sem custo por chamada.
- Qualidade inferior em manuscritos e imagens ruins; o servidor devolve a
  confiança média e o fluxo desvia para `low_quality` abaixo do limiar, em
  vez de seguir com texto ruim.
- A imagem Docker do OCR fica maior por causa do Tesseract e do OpenCV.
- Manuscrito com qualidade fica como bônus, com o resultado documentado.
