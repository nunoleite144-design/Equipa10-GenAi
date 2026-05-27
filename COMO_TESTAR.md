# Como correr e testar o Receiver GenAI

Pré-requisitos: Python 3.12 e `pip install -r requirements.txt`.

## Decodificar ficheiros JSON da Comunicação

O GenAI lê os ficheiros JSON que a Comunicação grava (tokens + metadados) e
reconstrói o áudio:

```powershell
python received_watcher.py --once --watch-dir <pasta_com_os_json> --output-dir output
```

Gera `output/<message_id>.wav` e `output/<message_id>_status.json`.

## Modo contínuo (integração ao vivo)

Com a Comunicação a gravar mensagens em tempo real, corre em modo contínuo — o
watcher fica a vigiar a pasta e reconstrói cada mensagem nova:

```powershell
python received_watcher.py --watch-dir <pasta_received> --output-dir output
```

## Testes

```powershell
python -m pytest -q
```

Os testes não carregam o modelo SemantiCodec.

## Notas

- Latência/qualidade: `--ddim-steps 50` (melhor qualidade) ou `25` (mais rápido).
- Se o JSON não trouxer `token_rate`/`semantic_vocab_size`, o decode usa 100/16384
  por defeito e avisa — estes valores devem ser combinados com a Equipa 9.
- O ruído de fundo na reconstrução é característico do SemantiCodec.
