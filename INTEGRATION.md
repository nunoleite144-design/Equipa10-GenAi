# Integration - GenAI Receiver

Este documento descreve o papel do repositório GenAI dentro do sistema global do projeto académico **Semantic Audio Communication using Generative AI - Receiver**.

O objetivo deste repositório não é implementar a WebApp nem todo o sistema MQTT. O objetivo principal é receber um payload SemantiCodec válido, reconstruir o áudio no Receiver e disponibilizar resultados claros para integração futura.

## Fluxo global

```text
Áudio original
  -> Equipa 9 / Transmitter
  -> SemantiCodec encode
  -> payload SemantiCodec
  -> Comunicação / MQTT
  -> Equipa 10 / GenAI Receiver
  -> SemantiCodec decode
  -> áudio reconstruído
  -> WebApp / demonstração
```

Em paralelo, a Equipa 9 pode gerar texto com Whisper:

```text
Áudio original
  -> Whisper
  -> transcrição textual
  -> WebApp / demonstração
```

A transcrição Whisper é apenas metadado auxiliar para visualização. Não substitui o payload SemantiCodec e não entra no processo de decode do GenAI.

## Papel do SemantiCodec

SemantiCodec é o codec principal do pipeline de áudio.

No lado Transmitter, SemantiCodec transforma o áudio original em tokens semânticos. Esses tokens são serializados num payload JSON e enviados ao Receiver.

No lado GenAI Receiver, o payload é validado, os tokens são recuperados e o SemantiCodec faz o decode para reconstruir o áudio.

O fluxo principal do GenAI é:

```text
payload SemantiCodec -> tokens -> SemantiCodec decode -> áudio reconstruído
```

## Papel do Whisper

Whisper é usado apenas para gerar texto complementar, por exemplo uma transcrição do áudio original.

Esse texto pode ser útil para a WebApp mostrar contexto durante a demonstração, mas não deve ser usado para reconstruir áudio.

O fluxo Whisper deve ficar separado do fluxo SemantiCodec:

```text
transcrição Whisper -> metadado visual -> WebApp
```

## Payload esperado pelo GenAI

O GenAI espera receber um payload JSON com tokens SemantiCodec.

Exemplo conceptual:

```json
{
  "message_id": "msg-001",
  "type": "semanticodec_tokens",
  "codec": "semanticodec",
  "token_rate": 100,
  "semantic_vocab_size": 16384,
  "sample_rate": 16000,
  "tokens_format": "torch_pt_base64",
  "tokens_shape": [1, 1000, 1],
  "tokens": "..."
}
```

Campos principais:

- `message_id`: identificador comum para correlacionar payload, áudio reconstruído, logs e possível texto Whisper.
- `type`: deve ser `semanticodec_tokens`.
- `codec`: deve ser `semanticodec`.
- `token_rate`: taxa de tokens usada pelo SemantiCodec. Valores atuais: `25`, `50` ou `100`.
- `semantic_vocab_size`: tamanho do vocabulário semântico. Valores atuais: `4096`, `8192`, `16384` ou `32768`.
- `sample_rate`: atualmente `16000`.
- `tokens_format`: atualmente `torch_pt_base64`.
- `tokens_shape`: forma dos tokens, útil para diagnóstico.
- `tokens`: tokens serializados em base64.

## Outputs produzidos pelo GenAI

O output principal do GenAI é um ficheiro de áudio reconstruído:

```text
output/<message_id>.wav
```

Também são relevantes os metadados de execução:

- `message_id`
- estado do processamento
- caminho do ficheiro de áudio reconstruído
- `codec`
- `token_rate`
- `semantic_vocab_size`
- `sample_rate`
- `tokens_shape`
- latência de decode
- parâmetros usados no decode, como `gain`, `ddim_steps` e `cfg_scale`
- erro, se o processamento falhar

## Informação futura para a WebApp

A WebApp deverá conseguir apresentar:

- estado do processamento: recebido, em validação, em decode, concluído ou erro;
- áudio reconstruído;
- `message_id`;
- transcrição Whisper, quando existir;
- latência de decode;
- codec usado;
- parâmetros técnicos relevantes para a demonstração;
- mensagens de erro simples e compreensíveis.

O texto Whisper deve ser associado ao mesmo `message_id`, mas deve continuar separado do processo de decode.

## Contrato conceptual: Comunicação -> GenAI

Responsabilidade da Comunicação:

- receber ou encaminhar o payload SemantiCodec produzido pela Equipa 9;
- garantir que o payload chega ao módulo GenAI;
- preservar `message_id`;
- distinguir mensagens de tokens SemantiCodec de mensagens de transcrição Whisper;
- decidir tópicos MQTT, QoS, ordem, retry e eventual fragmentação de mensagens grandes.

Mensagem esperada pelo GenAI:

```json
{
  "message_id": "msg-001",
  "type": "semanticodec_tokens",
  "codec": "semanticodec",
  "token_rate": 100,
  "semantic_vocab_size": 16384,
  "sample_rate": 16000,
  "tokens_format": "torch_pt_base64",
  "tokens": "..."
}
```

Mensagens de Whisper devem ser tratadas como outro tipo de mensagem, por exemplo:

```json
{
  "message_id": "msg-001",
  "type": "whisper_transcript",
  "text": "..."
}
```

O GenAI não deve depender desta mensagem para reconstruir áudio.

## Contrato conceptual: GenAI -> WebApp

Responsabilidade do GenAI:

- informar se o payload foi validado;
- informar se o decode foi concluido;
- disponibilizar o áudio reconstruído;
- expor metadados úteis para a demonstração;
- reportar erros de forma clara.

Exemplo conceptual de resultado:

```json
{
  "message_id": "msg-001",
  "status": "completed",
  "audio_file": "output/msg-001.wav",
  "codec": "semanticodec",
  "sample_rate": 16000,
  "token_rate": 100,
  "semantic_vocab_size": 16384,
  "decode_latency_ms": 12345
}
```

Exemplo conceptual de erro:

```json
{
  "message_id": "msg-001",
  "status": "failed",
  "stage": "payload_validation",
  "error_code": "invalid_payload",
  "error_message": "Payload missing required field(s): tokens"
}
```

## Dúvidas em aberto

A equipa ainda precisa de fechar:

- Que tópicos MQTT vão ser usados para tokens SemantiCodec e transcrição Whisper.
- Se o texto Whisper chega diretamente à WebApp ou se passa também pela Comunicação do Receiver.
- Se mensagens grandes com tokens vão precisar de fragmentação.
- Qual o limite máximo aceitável de tamanho do payload.
- Como a WebApp vai aceder ao ficheiro `.wav`: caminho local, endpoint HTTP, storage partilhado ou outro mecanismo.
- Que estados oficiais serão usados: `received`, `validating`, `decoding`, `completed`, `failed`, etc.
- Que formato final de erro deve ser partilhado entre departamentos.
- Se o formato `torch_pt_base64` é suficiente para a demo final ou se será necessário um formato mais neutro no futuro.
- Quem é responsável por limpar ficheiros antigos em `payloads/` e `output/`.

## Decisão atual

A decisão atual é manter SemantiCodec como caminho principal de reconstrução de áudio.

Whisper existe apenas como metadado textual auxiliar para a WebApp. Não é input do GenAI para decode e não substitui os tokens SemantiCodec.
