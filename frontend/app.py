import streamlit as st
import time
import random
import os
import subprocess
import sys
from pathlib import Path

st.set_page_config(page_title="Receiver - Equipa 10", layout="wide")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent
REPO_DIR     = FRONTEND_DIR.parent
GENAI_DIR    = REPO_DIR / "genai"
OUTPUT_DIR   = GENAI_DIR / "output"
PS1_SCRIPT   = GENAI_DIR / "run_genai_receiver.ps1"

# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------
if 'idioma' not in st.session_state:
    st.session_state.idioma = "Português 🇵🇹"
if 'conectado' not in st.session_state:
    st.session_state.conectado = False
if 'dados_prontos' not in st.session_state:
    st.session_state.dados_prontos = False
if 'latencia' not in st.session_state:
    st.session_state.latencia = 0
if 'watcher_pid' not in st.session_state:
    st.session_state.watcher_pid = None

# ---------------------------------------------------------------------------
# Helpers — watcher e ficheiros de output
# ---------------------------------------------------------------------------

def _watcher_vivo() -> bool:
    """Verifica se o processo watcher ainda está ativo (Windows e Unix)."""
    pid = st.session_state.watcher_pid
    if pid is None:
        return False
    try:
        # Unix: os.kill com sinal 0 apenas verifica se o processo existe
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        st.session_state.watcher_pid = None
        return False
    except OSError:
        # Windows não suporta os.kill com sinal 0 — usa tasklist
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=3,
            )
            return str(pid) in result.stdout
        except Exception:
            return False


def _iniciar_watcher() -> None:
    """Lança o watcher em background.

    Windows -> corre o run_genai_receiver.ps1 via PowerShell.
    Linux/Mac -> corre o received_watcher.py diretamente com Python do venv (ou PATH).
    """
    import platform
    if platform.system() == "Windows":
        cmd = [
            "powershell",
            "-ExecutionPolicy", "Bypass",
            "-File", str(PS1_SCRIPT),
        ]
    else:
        # Ubuntu / macOS: usa o Python do venv se existir, senão sys.executable
        venv_python = REPO_DIR / "venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else sys.executable
        cmd = [
            python,
            str(GENAI_DIR / "received_watcher.py"),
            "--watch-dir", str(REPO_DIR / "comunicacao" / "received"),
            "--output-dir", str(GENAI_DIR / "output"),
        ]
    proc = subprocess.Popen(cmd, cwd=str(GENAI_DIR))
    st.session_state.watcher_pid = proc.pid


def _ficheiro_audio_recente() -> str | None:
    """Devolve o caminho do .wav mais recente em genai/output/, ou None."""
    if not OUTPUT_DIR.exists():
        return None
    wavs = list(OUTPUT_DIR.glob("*.wav"))
    if not wavs:
        return None
    return str(max(wavs, key=lambda p: p.stat().st_mtime))


def _verificar_dados_prontos() -> None:
    """Atualiza session_state.dados_prontos com base nos ficheiros reais."""
    st.session_state.dados_prontos = _ficheiro_audio_recente() is not None

# ---------------------------------------------------------------------------
# Sidebar — idioma
# ---------------------------------------------------------------------------
st.sidebar.title("Definições / Settings")
st.session_state.idioma = st.sidebar.radio(
    "Escolhe o idioma / Choose language:",
    ["Português 🇵🇹", "English 🇬🇧"]
)

# ---------------------------------------------------------------------------
# Strings PT / EN  (UI inalterada)
# ---------------------------------------------------------------------------
if st.session_state.idioma == "Português 🇵🇹":
    titulo           = "📡 Recetor - Equipa 10"
    btn_conectar     = "🔌 Conectar ao Transmissor"
    msg_conectar_sucesso = "Sinal estabelecido de forma segura."
    btn_reconstruir  = "🧠 Iniciar Reconstrução GenAI"
    msg_recon_sucesso = "Áudio reconstruído com sucesso!"
    lbl_latencia     = "Latência de Rede"
    lbl_pacotes      = "Integridade do Sinal"
    msg_espera       = "A aguardar ligação do recetor..."
    titulo_out       = "🎵 Áudio Reconstruído"
else:
    titulo           = "📡 Receiver - Team 10"
    btn_conectar     = "🔌 Connect to Transmitter"
    msg_conectar_sucesso = "Signal established securely."
    btn_reconstruir  = "🧠 Start GenAI Reconstruction"
    msg_recon_sucesso = "Audio successfully reconstructed!"
    lbl_latencia     = "Network Latency"
    lbl_pacotes      = "Signal Integrity"
    msg_espera       = "Waiting for receiver connection..."
    titulo_out       = "🎵 Reconstructed Audio"

# ---------------------------------------------------------------------------
# UI principal (layout inalterado)
# ---------------------------------------------------------------------------
st.title(titulo)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Conexão" if st.session_state.idioma == "Português 🇵🇹" else "1. Connection")
    if st.button(btn_conectar, use_container_width=True):
        with st.spinner("..."):
            time.sleep(1.5)
            st.session_state.conectado = True
            st.session_state.latencia = random.randint(12, 35)
            # Lança o watcher (PS1) se ainda não estiver ativo
            if not _watcher_vivo():
                _iniciar_watcher()
            # Verifica desde já se há áudio disponível
            _verificar_dados_prontos()
        st.success(msg_conectar_sucesso)

# Se já conectado, atualiza o estado dos dados a cada render
if st.session_state.conectado:
    _verificar_dados_prontos()

with col2:
    st.subheader("2. Telemetria" if st.session_state.idioma == "Português 🇵🇹" else "2. Telemetry")
    if st.session_state.conectado:
        metrica_col1, metrica_col2 = st.columns(2)
        metrica_col1.metric(label=lbl_latencia, value=f"{st.session_state.latencia} ms", delta="-2 ms", delta_color="inverse")
        metrica_col2.metric(label=lbl_pacotes, value="100%", delta="OK")
    else:
        st.info(msg_espera)

st.write("")
st.write("")

if st.session_state.dados_prontos:
    if st.button(btn_reconstruir, type="primary", use_container_width=True):
        barra_progresso = st.progress(0)
        for percentagem in range(100):
            time.sleep(0.03)
            barra_progresso.progress(percentagem + 1)

        st.success(msg_recon_sucesso)

        st.divider()
        st.subheader(titulo_out)

        # Áudio real gerado pelo watcher GenAI
        audio_path = _ficheiro_audio_recente()
        if audio_path:
            st.audio(audio_path, format="audio/wav")
        else:
            st.warning("Nenhum ficheiro de áudio encontrado em genai/output/")
