# GenAI Receiver - SemantiCodec Demo

Demo da subequipa GenAI para o projeto **Semantic Audio Communication using Generative AI**.

O caminho principal usa **SemantiCodec** para preservar voz/timbre:

```text
Áudio original -> SemantiCodec tokens -> JSON/base64 -> SemantiCodec decode -> áudio reconstruído
```

## 1. Instalar dependências

```bash
python3 -m pip install -r requirements.txt
```

Na primeira execução, o SemantiCodec pode descarregar checkpoints/modelos grandes. Por defeito, eles ficam fora da pasta do repositório/iCloud:

```text
~/.cache/genai-semanticodec/pretrained
```

Para apagar os modelos e libertar espaço:

```bash
rm -rf ~/.cache/genai-semanticodec
```

## 2. Criar payload no lado Transmitter

Usa um áudio curto, de preferência `.wav` com 5 a 10 segundos.

```bash
python3 semantic_transmitter_demo.py input.wav --message-id msg-001
```

Resultado esperado:

```text
payloads/msg-001.json
```

Esse JSON é o pacote que a subequipa MQTT pode transmitir.

## 3. Reconstruir áudio no lado Receiver

```bash
python3 semantic_receiver_demo.py payloads/msg-001.json
```

Resultado esperado:

```text
output/msg-001.wav
```

Se o áudio reconstruído ficar baixo, ajusta o ganho:

```bash
python3 semantic_receiver_demo.py payloads/msg-001.json --gain 1.8
```

Para testes mais rápidos, podes reduzir os passos do decoder:

```bash
python3 semantic_receiver_demo.py payloads/msg-001.json --ddim-steps 25
```

Menos passos é mais rápido, mas pode reduzir a qualidade.

## 4. Comando único para teste local

```bash
python3 semantic_demo.py input.wav --message-id msg-001
```

Isto corre o transmitter e o receiver de seguida.

## 5. Correr testes

Os testes unitários atuais validam o contrato do payload SemantiCodec sem carregar modelos nem descarregar checkpoints.

```bash
python3 -m pytest
```

## Papel no sistema global

Este repositório pertence ao departamento GenAI do Receiver. O input principal do GenAI é o payload SemantiCodec com tokens. O texto gerado por Whisper, quando existir, é apenas metadado auxiliar para visualização na WebApp e não entra no processo de decode.

Para detalhes de integração com Comunicação e WebApp, ver `INTEGRATION.md`.

## Contrato JSON para MQTT

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

## Ficheiros principais

- `semantic_transmitter_demo.py`: cria tokens e payload JSON a partir de áudio.
- `semantic_receiver.py`: API reutilizável para reconstruir áudio a partir de payload em memória ou ficheiro JSON.
- `semantic_receiver_demo.py`: CLI de demo que reconstrói áudio a partir do payload.
- `semantic_demo.py`: corre encode e decode numa só chamada.
- `semantic_payload.py`: validação e serialização dos tokens.
- `semanticodec/`: cópia vendorizada do pacote SemantiCodec.
- `vendor/semanticodec_source/`: README e LICENSE originais do SemantiCodec.

## Erros comuns

- Áudio não encontrado: confirma o caminho passado ao `semantic_transmitter_demo.py`.
- Dependências Torch/Torchaudio falham: instala uma versão compatível com o sistema.
- Checkpoint demora: a primeira execução pode descarregar modelos do Hugging Face.
- iCloud a subir ficheiros: os modelos antigos podem estar em `pretrained/`; apaga essa pasta se já não precisares.
- Payload inválido: confirma `type`, `token_rate`, `semantic_vocab_size` e `tokens_format`.
