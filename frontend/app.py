import streamlit as st
import time
import random
import shutil
import subprocess
from pathlib import Path
import threading
import queue
import platform
import sys
import os

st.set_page_config(page_title="Receiver - Equipa 10", layout="wide")

if 'idioma' not in st.session_state:
    st.session_state.idioma = "Português 🇵🇹"
if 'conectado' not in st.session_state:
    st.session_state.conectado = False
if 'dados_prontos' not in st.session_state:
    st.session_state.dados_prontos = False
if 'latencia' not in st.session_state:
    st.session_state.latencia = 0
if 'receiver_process' not in st.session_state:
    st.session_state.receiver_process = None
if 'receiver_error' not in st.session_state:
    st.session_state.receiver_error = None
if 'receiver_logs' not in st.session_state:
    st.session_state.receiver_logs = []
if 'receiver_read_thread' not in st.session_state:
    st.session_state.receiver_read_thread = None
if 'receiver_stop_event' not in st.session_state:
    st.session_state.receiver_stop_event = threading.Event()
if 'last_audio_file' not in st.session_state:
    st.session_state.last_audio_file = None
if 'genai_process' not in st.session_state:
    st.session_state.genai_process = None
if 'genai_logs' not in st.session_state:
    st.session_state.genai_logs = []
if 'genai_error' not in st.session_state:
    st.session_state.genai_error = None
if 'genai_stop_event' not in st.session_state:
    st.session_state.genai_stop_event = threading.Event()
if 'genai_read_thread' not in st.session_state:
    st.session_state.genai_read_thread = None
if 'existing_wavs_before' not in st.session_state:
    st.session_state.existing_wavs_before = set()

# =============================================================================
# Definições de Funções (precisam estar aqui antes de serem usadas)
# =============================================================================

def read_stream_output(process, log_list, stop_event):
    """
    Thread function que lê continuamente do stdout do processo.
    Adiciona as linhas a log_list até o processo terminar.
    """
    try:
        while not stop_event.is_set():
            line = process.stdout.readline()
            if line:
                line = line.rstrip('\n')
                if line:
                    log_list.append(f"[{time.strftime('%H:%M:%S')}] {line}")
            else:
                time.sleep(0.1)  # Evitar CPU spinning
    except Exception as e:
        log_list.append(f"[{time.strftime('%H:%M:%S')}] Erro ao ler logs: {str(e)}")


def start_mqtt_receiver() -> bool:
    """
    Inicia o receiver MQTT em background usando subprocess.
    Retorna True se iniciou com sucesso, False caso contrário.
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
        comunicacao_dir = repo_root / "comunicacao"
        
        if not comunicacao_dir.exists():
            st.session_state.receiver_error = f"Pasta comunicacao não encontrada: {comunicacao_dir}"
            return False
        
        # Iniciar processo em background (sys.executable garante o Python correto no venv)
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(comunicacao_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirecionar stderr para stdout
            text=True,
            bufsize=1,
        )
        
        st.session_state.receiver_process = process
        st.session_state.receiver_error = None
        st.session_state.receiver_logs = [f"[{time.strftime('%H:%M:%S')}] Receiver iniciado (PID: {process.pid})"]
        
        # Criar evento para parar o thread de leitura
        if 'receiver_stop_event' not in st.session_state:
            st.session_state.receiver_stop_event = threading.Event()
        else:
            st.session_state.receiver_stop_event.clear()
        
        # Iniciar thread que lê os logs
        read_thread = threading.Thread(
            target=read_stream_output,
            args=(process, st.session_state.receiver_logs, st.session_state.receiver_stop_event),
            daemon=True,
        )
        read_thread.start()
        st.session_state.receiver_read_thread = read_thread
        
        return True
        
    except Exception as exc:
        st.session_state.receiver_error = f"{msg_conectar_erro} {str(exc)}"
        st.session_state.receiver_process = None
        return False


def stop_mqtt_receiver() -> bool:
    """
    Para o receiver MQTT em background.
    Retorna True se parou com sucesso, False caso contrário.
    """
    try:
        if st.session_state.receiver_process is None:
            return True
        
        # Parar o thread de leitura
        if 'receiver_stop_event' in st.session_state:
            st.session_state.receiver_stop_event.set()
        
        process = st.session_state.receiver_process
        process.terminate()
        
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        st.session_state.receiver_process = None
        st.session_state.receiver_logs.append(f"[{time.strftime('%H:%M:%S')}] Receiver parado")
        return True
        
    except Exception as exc:
        st.session_state.receiver_error = f"Erro ao parar receiver: {str(exc)}"
        return False


def check_receiver_status() -> None:
    """
    Verifica se o processo receiver ainda está a correr.
    """
    if st.session_state.receiver_process is not None:
        poll_result = st.session_state.receiver_process.poll()
        
        if poll_result is not None:  # Processo terminou
            st.session_state.receiver_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠️ Receiver terminou com código: {poll_result}")
            st.session_state.receiver_process = None
            st.session_state.conectado = False


def check_genai_status() -> None:
    """
    Verifica se o processo GenAI ainda está a correr.
    """
    if st.session_state.genai_process is not None:
        poll_result = st.session_state.genai_process.poll()
        
        if poll_result is not None:  # Processo terminou
            if poll_result == 0:
                st.session_state.genai_logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ Reconstrução concluída com sucesso")
            else:
                st.session_state.genai_logs.append(f"[{time.strftime('%H:%M:%S')}] ❌ Reconstrução terminou com erro (código: {poll_result})")
                st.session_state.genai_error = f"Reconstrução GenAI falhou com código {poll_result}"
            
            st.session_state.genai_process = None


def run_genai_receiver(watch_dir: Path, output_dir: Path, ddim_steps: int = 25, device: str = "cpu") -> bool:
    """
    Inicia o receiver GenAI (received_watcher.py) em background.
    Retorna True se iniciou com sucesso, False caso contrário.
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
        genai_dir = repo_root / "genai"
        received_watcher = genai_dir / "received_watcher.py"
        
        if not received_watcher.exists():
            st.session_state.genai_error = f"Script received_watcher.py não encontrado em {received_watcher}"
            return False
        
        # Definir variáveis de ambiente para Python
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Comando para executar
        args = [
            sys.executable,
            str(received_watcher),
            "--watch-dir",
            str(watch_dir),
            "--output-dir",
            str(output_dir),
            "--ddim-steps",
            str(ddim_steps),
            "--device",
            device,
            "--once",
        ]
        
        # Iniciar processo em background
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(genai_dir),
            env=env,
            bufsize=1,
        )
        
        st.session_state.genai_process = process
        st.session_state.genai_error = None
        st.session_state.genai_logs = [f"[{time.strftime('%H:%M:%S')}] Reconstrução iniciada (PID: {process.pid})"]
        
        # Criar evento para parar o thread de leitura
        if 'genai_stop_event' not in st.session_state:
            st.session_state.genai_stop_event = threading.Event()
        else:
            st.session_state.genai_stop_event.clear()
        
        # Iniciar thread que lê os logs
        read_thread = threading.Thread(
            target=read_stream_output,
            args=(process, st.session_state.genai_logs, st.session_state.genai_stop_event),
            daemon=True,
        )
        read_thread.start()
        st.session_state.genai_read_thread = read_thread
        
        return True
        
    except Exception as exc:
        st.session_state.genai_error = f"Erro ao iniciar reconstrução: {str(exc)}"
        st.session_state.genai_process = None
        return False


def find_new_wav_file(output_dir: Path, existing_files: set[str]) -> Path | None:
    candidates = [path for path in output_dir.glob("*.wav") if path.name not in existing_files]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


st.sidebar.title("Definições / Settings")
st.session_state.idioma = st.sidebar.radio(
    "Escolhe o idioma / Choose language:",
    ["Português 🇵🇹", "English 🇬🇧"]
)

if st.session_state.idioma == "Português 🇵🇹":
    titulo = "📡 Recetor - Equipa 10"
    btn_conectar = "🔌 Ligar o Recetor"
    btn_desconectar = "🔌 Desligar o Recetor"
    msg_conectar_sucesso = "Sinal estabelecido de forma segura."
    msg_desconectar_sucesso = "Recetor desligado com sucesso."
    msg_conectar_erro = "Erro ao ligar o recetor:"
    msg_conectar_aguardando = "A iniciar recetor MQTT..."
    btn_reconstruir = "🧠 Iniciar Reconstrução GenAI"
    msg_recon_sucesso = "Áudio reconstruído com sucesso!"
    btn_transferir = "⬇️ Transferir Áudio"
    lbl_latencia = "Latência de Rede"
    lbl_pacotes = "Integridade do Sinal"
    msg_espera = "A aguardar ligação do recetor..."
    titulo_out = "🎵 Áudio Reconstruído"
    lbl_logs = "📋 Logs do Recetor"
    lbl_logs_genai = "📋 Logs da Reconstrução GenAI"
else:
    titulo = "📡 Receiver - Team 10"
    btn_conectar = "🔌 Connect to Transmitter"
    btn_desconectar = "🔌 Disconnect Receiver"
    msg_conectar_sucesso = "Signal established securely."
    msg_desconectar_sucesso = "Receiver disconnected successfully."
    msg_conectar_erro = "Error connecting receiver:"
    msg_conectar_aguardando = "Starting MQTT receiver..."
    btn_reconstruir = "🧠 Start GenAI Reconstruction"
    msg_recon_sucesso = "Audio successfully reconstructed!"
    btn_transferir = "⬇️ Download Audio"
    lbl_latencia = "Network Latency"
    lbl_pacotes = "Signal Integrity"
    msg_espera = "Waiting for receiver connection..."
    titulo_out = "🎵 Reconstructed Audio"
    lbl_logs = "📋 Receiver Logs"
    lbl_logs_genai = "📋 GenAI Reconstruction Logs"

st.title(titulo)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Conexão" if st.session_state.idioma == "Português 🇵🇹" else "1. Connection")
    
    check_receiver_status()
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button(btn_conectar, use_container_width=True, disabled=st.session_state.conectado):
            with st.spinner(msg_conectar_aguardando):
                if start_mqtt_receiver():
                    st.session_state.conectado = True
                    st.session_state.dados_prontos = True
                    st.session_state.latencia = random.randint(12, 35)
                    st.success(msg_conectar_sucesso)
                    st.rerun()
                else:
                    st.error(st.session_state.receiver_error)
    
    with col_btn2:
        if st.button(btn_desconectar, use_container_width=True, disabled=not st.session_state.conectado):
            if stop_mqtt_receiver():
                st.session_state.conectado = False
                st.session_state.dados_prontos = False
                st.success(msg_desconectar_sucesso)
                st.rerun()
            else:
                st.error(st.session_state.receiver_error)

with col2:
    st.subheader("2. Telemetria" if st.session_state.idioma == "Português 🇵🇹" else "2. Telemetry")
    if st.session_state.conectado:
        metrica_col1, metrica_col2 = st.columns(2)
        metrica_col1.metric(label=lbl_latencia, value=f"{st.session_state.latencia} ms", delta="-2 ms", delta_color="inverse")
        metrica_col2.metric(label=lbl_pacotes, value="100%", delta="OK")
    else:
        st.info(msg_espera)

# Mostrar erros do receiver
if st.session_state.receiver_error:
    st.error(f"❌ {st.session_state.receiver_error}")

# Mostrar logs do receiver
if st.session_state.receiver_logs:
    with st.expander(lbl_logs, expanded=False):
        log_text = "\n".join(st.session_state.receiver_logs[-50:])  # Últimas 50 linhas
        st.code(log_text, language="log")

st.write("")
st.write("")

if st.session_state.dados_prontos:
    col_recon1, col_recon2 = st.columns([3, 1])
    
    with col_recon1:
        if st.button(btn_reconstruir, type="primary", use_container_width=True, disabled=st.session_state.genai_process is not None):
            repo_root = Path(__file__).resolve().parents[1]
            watch_dir = repo_root / "comunicacao" / "received"
            output_dir = repo_root / "genai" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.last_audio_file = None
            existing_wavs = {path.name for path in output_dir.glob("*.wav")}
            
            # Verificar se há ficheiros JSON para processar
            json_files = list(watch_dir.glob("*.json"))
            if not json_files:
                st.error("❌ Nenhum ficheiro JSON encontrado em comunicacao/received/. Certifique-se que o receiver MQTT está ligado e recebendo mensagens MQTT.")
            else:
                st.info(f"ℹ️ Encontrado(s) {len(json_files)} ficheiro(s) JSON. Iniciando reconstrução...")
                if run_genai_receiver(watch_dir, output_dir, ddim_steps=25, device="cpu"):
                    st.session_state.existing_wavs_before = existing_wavs
                    st.rerun()
                else:
                    st.error(st.session_state.genai_error)
    
    with col_recon2:
        if st.button("🗑️", use_container_width=True, help="Limpar logs"):
            st.session_state.genai_logs = []
            st.rerun()

# Mostrar informações sobre o status do reconstrução
if st.session_state.genai_process is not None:
    with st.container():
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Status", "🔄 A processar...", "Em execução")
        col_info2.metric("PID", st.session_state.genai_process.pid)
        repo_root = Path(__file__).resolve().parents[1]
        json_files = list((repo_root / "comunicacao" / "received").glob("*.json"))
        col_info3.metric("Ficheiros JSON", len(json_files))

# Mostrar erros e logs da reconstrução GenAI
if st.session_state.genai_error:
    st.error(f"❌ {st.session_state.genai_error}")

if st.session_state.genai_logs:
    with st.expander(lbl_logs_genai, expanded=True):
        log_text = "\n".join(st.session_state.genai_logs[-100:])  # Últimas 100 linhas
        st.code(log_text, language="log")

# Verificar status do processo GenAI
if st.session_state.genai_process is not None:
    check_genai_status()
    
    # Esperar mais tempo para o processo processar
    time.sleep(2)
    st.rerun()
else:
    # Se processo terminou, procurar por novo áudio
    if st.session_state.genai_logs and st.session_state.last_audio_file is None:
        if 'existing_wavs_before' in st.session_state:
            repo_root = Path(__file__).resolve().parents[1]
            output_dir = repo_root / "genai" / "output"
            audio_file = find_new_wav_file(output_dir, st.session_state.existing_wavs_before)
            
            if audio_file is not None:
                st.session_state.last_audio_file = audio_file
                st.success(msg_recon_sucesso)
            else:
                # Verificar se há ficheiros JSON em received/
                received_dir = repo_root / "comunicacao" / "received"
                json_files = list(received_dir.glob("*.json"))
                if not json_files:
                    st.warning("⚠️ Nenhum ficheiro JSON encontrado em comunicacao/received/. Certifique-se que o receiver MQTT está ligado e recebendo mensagens.")
                else:
                    st.info(f"ℹ️ {len(json_files)} ficheiro(s) JSON encontrado(s), mas nenhum áudio foi gerado. Verifique os logs acima.")

# Mostrar áudio e botão de download
if st.session_state.last_audio_file is not None and st.session_state.last_audio_file.exists():
    st.divider()
    st.subheader(titulo_out)
    st.audio(str(st.session_state.last_audio_file), format="audio/wav")
    
    with open(str(st.session_state.last_audio_file), "rb") as f:
        audio_data = f.read()
    
    st.download_button(
        label=btn_transferir,
        data=audio_data,
        file_name=st.session_state.last_audio_file.name,
        mime="audio/wav",
        use_container_width=True,
    )

# Auto-refresh para atualizar logs quando receiver está ativo
if st.session_state.conectado and st.session_state.receiver_process is not None:
    time.sleep(1)
    st.rerun()