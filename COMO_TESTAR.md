# Como correr e testar o Receiver GenAI

Pré-requisitos: Python 3.12 e `pip install -r requirements.txt`.

## Estrutura de pastas (deploy)

O launcher assume esta estrutura, com o GenAI e a Comunicação como pastas irmãs:

```text
projeto/
  <pasta GenAI>/        <- este repo; corre aqui o launcher
  comunicacao/received/ <- a Comunicação grava os JSON aqui (o GenAI vigia)
  webapp/
```

É portável (não depende da máquina): o caminho é resolvido relativamente à
localização do repo. Se a tua estrutura for diferente, passa `-WatchDir`.

## Modo automático (recomendado)

O GenAI vigia `../comunicacao/received` e reconstrói o áudio de cada mensagem nova
automaticamente. Arranca com um comando:

```powershell
powershell -ExecutionPolicy Bypass -File run_genai_receiver.ps1
```

Ajusta com `-WatchDir` se a pasta for noutro sítio (ou `-DdimSteps 25` p/ mais rápido):

```powershell
powershell -ExecutionPolicy Bypass -File run_genai_receiver.ps1 -WatchDir "D:\...\received"
```

Para cada `received/<...>.json`, gera:
- `output/<message_id>.wav` — áudio reconstruído
- `output/<message_id>_status.json` — estado (`completed` / `failed`)

e move o JSON tratado para `received/processed/` (ou `received/failed/` em erro), por
isso `received/` fica sempre só com mensagens por processar e um reinício não
reprocessa o que já foi feito. (Ctrl+C para parar.)

## Sem o launcher (chamada direta)

Uma só passagem sobre os ficheiros existentes (e sai):

```powershell
python received_watcher.py --once --watch-dir <pasta_com_os_json> --output-dir output
```

Modo contínuo (equivalente ao launcher, mas sem as variáveis de ambiente do Windows):

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
  Em CPU, ~30 s por mensagem com 50 passos.
- O decode usa `token_rate=100` / `semantic_vocab_size=16384` por defeito (acordado
  com a Equipa 9, que não envia estes campos no JSON). Um JSON pode incluí-los para
  os sobrepor.
- Os `tokens` podem vir achatados (`[s,a,s,a,...]`) ou aninhados (`[[s,a],...]`); o
  adaptador trata ambos.
- O ruído de fundo na reconstrução é característico do SemantiCodec.
