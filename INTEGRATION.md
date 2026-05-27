# Integração - GenAI Receiver

Este documento descreve o papel do repositório GenAI dentro do sistema global do
projeto académico **Semantic Audio Communication using Generative AI - Receiver** e
o contrato de integração com as outras subequipas.

O objetivo deste repositório não é implementar a WebApp nem todo o sistema MQTT. É
receber os tokens SemantiCodec, reconstruir o áudio no Receiver e disponibilizar
resultados claros (`.wav` + estado) para a WebApp.

## Fluxo global

```text
Áudio original
  -> Equipa 9 / Transmitter
  -> SemantiCodec encode
  -> payload com tokens
  -> Comunicação / MQTT
  -> Equipa 10 / GenAI Receiver
  -> SemantiCodec decode
  -> áudio reconstruído
  -> WebApp / demonstração
```

Em paralelo, a Equipa 9 gera texto com Whisper:

```text
Áudio original -> Whisper -> transcrição textual -> WebApp / demonstração
```

A transcrição Whisper é apenas metadado auxiliar para visualização. Não substitui os
tokens SemantiCodec e não entra no processo de decode do GenAI.

## O GenAI é SemantiCodec

SemantiCodec é o codec principal do pipeline de áudio: tokens semânticos -> áudio
direto, **sem TTS e sem reconstrução de texto**.

No lado Transmitter, transforma o áudio em tokens semânticos. No lado GenAI Receiver,
os tokens são validados e o SemantiCodec faz o decode para reconstruir o áudio.

```text
tokens SemantiCodec -> SemantiCodec decode -> áudio reconstruído
```

## Componentes (lado GenAI)

- `semantic_payload.py` — contrato/serialização canónica (tokens em base64 de tensor
  torch); usado pelas CLIs de teste local (`semantic_demo.py`, etc.).
- `semantic_receiver.py` — `decode_tokens(tensor, ...)` (core de decode partilhado),
  `decode_payload(...)` (formato canónico), `load_model(...)` (construir o modelo
  uma vez e reutilizar).
- `comunicacao_adapter.py` — converte o JSON da Comunicação (tokens em lista `[N,2]`)
  num tensor `[1,N,2]` e chama o core. Ignora o campo `text` (Whisper).
- `received_watcher.py` — vigia a pasta `received/`, reutiliza o modelo entre
  mensagens e escreve `output/<id>.wav` + `output/<id>_status.json`.

## Handoff Comunicação -> GenAI (por ficheiro)

A Comunicação trata do MQTT e da desencriptação e **grava um JSON por mensagem** em
`received/`. O GenAI vigia essa pasta e reconstrói o áudio; não subscreve MQTT
diretamente.

## Contrato de entrada (Comunicação -> GenAI)

Cada ficheiro em `received/` é um JSON com os tokens já em claro:

```json
{
  "message_id": "<uuid>",
  "tokens": [[semantico, acustico], "..."],
  "token_rate": 100,
  "semantic_vocab_size": 16384,
  "sample_rate": 16000,
  "text": "transcrição Whisper (opcional, só para a WebApp)"
}
```

| Campo | Obrigatório | Notas |
|---|---|---|
| `message_id` | sim | preservado no nome do `.wav` e no estado |
| `tokens` | sim | lista 2D `[N,2]` (semântico, acústico) |
| `token_rate` | sim | 25, 50 ou 100 — tem de ser igual ao encode da Equipa 9 |
| `semantic_vocab_size` | sim | 4096, 8192, 16384 ou 32768 — igual ao encode |
| `sample_rate` | opcional | por defeito 16000 |
| `text` | opcional | transcrição Whisper; não entra na reconstrução |

`token_rate` e `semantic_vocab_size` têm de circular por toda a cadeia: Equipa 9
(encode) -> Comunicação (preserva no JSON) -> GenAI. Se faltarem no JSON, o decode usa
100/16384 por defeito e avisa, mas um valor errado produz áudio inintelígivel.

> Nota: para ferramentas e testes locais existe também o formato canónico
> (`semantic_payload.py`), em que os tokens vão como tensor torch em base64
> (`tokens_format: torch_pt_base64`, com os campos `type`/`codec`). É equivalente;
> o `comunicacao_adapter.py` é que trata da forma em lista enviada pela Comunicação.

## Saída (GenAI -> WebApp)

- `output/<message_id>.wav`
- `output/<message_id>_status.json`

Estados válidos: `received`, `validating`, `decoding`, `completed`, `failed`.

Exemplo de estado de sucesso:

```json
{
  "message_id": "msg-001",
  "status": "completed",
  "audio_file": "output/msg-001.wav",
  "codec": "semanticodec",
  "sample_rate": 16000,
  "token_rate": 100,
  "semantic_vocab_size": 16384,
  "tokens_shape": [1, 504, 2],
  "decode_latency_ms": 12345,
  "ddim_steps": 50,
  "gain": 1.5
}
```

Exemplo de erro:

```json
{
  "message_id": "msg-001",
  "status": "failed",
  "stage": "payload_validation",
  "error_code": "invalid_payload",
  "error_message": "Payload missing 'message_id'."
}
```

## Como correr

Ver `COMO_TESTAR.md`.

## Dúvidas em aberto / pendências

- Equipa 9 + Comunicação: incluir `token_rate` e `semantic_vocab_size` no JSON.
- Alinhar a versão do `paho-mqtt` (o módulo de Comunicação usa a API 1.x).
- Mover a chave AES de `config.py` (módulo de Comunicação) para variável de ambiente / `.env`.
- Que tópicos MQTT vão ser usados para tokens SemantiCodec e transcrição Whisper.
- Como a WebApp vai aceder ao `.wav`: caminho local, endpoint HTTP, storage partilhado.
- Limite máximo de tamanho do payload e eventual fragmentação de mensagens grandes.
- Formato de erro a partilhar entre subequipas.
