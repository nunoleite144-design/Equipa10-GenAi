# run_genai_receiver.ps1 - Receiver GenAI (Equipa 10) em modo automatico.
#
# Vigia a pasta onde a Comunicacao grava os JSON desencriptados e reconstroi o
# audio: received/*.json -> output/<message_id>.wav (+ _status.json). Os JSON
# tratados passam para received/processed/ (ou received/failed/ em erro).
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File run_genai_receiver.ps1
#   powershell -ExecutionPolicy Bypass -File run_genai_receiver.ps1 -DdimSteps 25
#   powershell -ExecutionPolicy Bypass -File run_genai_receiver.ps1 -WatchDir "D:\caminho\received"
#
# Requer Python 3.12 e as dependencias. Ctrl+C para parar; o modelo carrega na
# primeira mensagem e fica em cache.
param(
    [string]$WatchDir  = "",
    [string]$OutputDir = "output",
    [int]   $DdimSteps = 50,
    [string]$Device    = "cpu"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path

# Estrutura esperada: projeto\{<esta pasta GenAI>, comunicacao\received, webapp}.
# Resolve a pasta irma comunicacao\received (com ou sem cedilha). Para outra
# localizacao, passa -WatchDir "caminho\completo\received".
if (-not $WatchDir) {
    $projeto    = Split-Path -Parent $repo
    $cedilha    = "comunica" + [char]0x00E7 + [char]0x00E3 + "o"
    $candidatos = @("comunicacao", $cedilha) | ForEach-Object { Join-Path $projeto (Join-Path $_ "received") }
    $WatchDir   = $candidatos | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $WatchDir) { $WatchDir = $candidatos[0] }
}

# O Torch no Windows precisa destas variaveis (evita o conflito de OpenMP).
$env:KMP_DUPLICATE_LIB_OK            = "TRUE"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING                = "utf-8"

# Prioridade: 1) venv do projeto  2) Python 3.12 do sistema  3) python do PATH
$venv = Join-Path (Split-Path -Parent $repo) "venv\Scripts\python.exe"
$py312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if     (Test-Path $venv) { $py = $venv }
elseif (Test-Path $py312) { $py = $py312 }
else                      { $py = "python" }
Write-Host "  Python        : $py"

Write-Host "Receiver GenAI (SemantiCodec) - modo automatico"
Write-Host "  Pasta vigiada : $WatchDir"
Write-Host "  Saida (.wav)  : $OutputDir"
Write-Host "  ddim-steps    : $DdimSteps    device: $Device"
Write-Host "  (Ctrl+C para parar)"
Write-Host ""

Set-Location $repo
& $py received_watcher.py --watch-dir $WatchDir --output-dir $OutputDir --ddim-steps $DdimSteps --device $Device
