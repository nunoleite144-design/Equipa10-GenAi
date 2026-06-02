# Como correr e testar o Receiver GenAI

Pré-requisitos: Python 3.12.

## Preparação (primeira vez)

1. Clonar o repositório e garantir que se está no `main`:

   ```powershell
   git clone https://github.com/nunoleite144-design/Equipa10-GenAi.git
   cd Equipa10-GenAi
   git checkout main
   ```

2. Instalar as dependências (demora — está a instalar o Torch):

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Descarregar o modelo SemantiCodec **uma vez** (~2,3 GB do Hugging Face). Copiar as
   duas linhas, colar no PowerShell e esperar até aparecer `Pronto`:

   ```powershell
   $env:KMP_DUPLICATE_LIB_OK="TRUE"
   python -c "from semantic_receiver import load_model; load_model(token_rate=100, semantic_vocab_size=16384); print('Pronto')"
   ```

   Depois disto fica em cache e o decode passa a ser rápido. Convém fazê-lo com
   antecedência, para não estar a descarregar durante a demonstração.

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
