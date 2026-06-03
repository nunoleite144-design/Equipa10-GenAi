import streamlit as st
import time
import random
import os

st.set_page_config(page_title="Receiver - Equipa 10", layout="wide")

# Caminhos para ler o resultado do Vigilante (GenAI)
PASTA_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PASTA_BASE, "output")

if 'idioma' not in st.session_state:
    st.session_state.idioma = "Português 🇵🇹"
if 'conectado' not in st.session_state:
    st.session_state.conectado = False
if 'dados_prontos' not in st.session_state:
    st.session_state.dados_prontos = False
if 'latencia' not in st.session_state:
    st.session_state.latencia = 0

st.sidebar.title("Definições / Settings")
st.session_state.idioma = st.sidebar.radio(
    "Escolhe o idioma / Choose language:",
    ["Português 🇵🇹", "English 🇬🇧"]
)

if st.session_state.idioma == "Português 🇵🇹":
    titulo = "📡 Recetor - Equipa 10"
    btn_conectar = "🔌 Conectar ao Transmissor"
    msg_conectar_sucesso = "Sinal estabelecido de forma segura. À escuta..."
    btn_reconstruir = "🧠 Iniciar Reconstrução GenAI"
    msg_recon_sucesso = "Áudio reconstruído com sucesso!"
    lbl_latencia = "Latência de Rede"
    lbl_pacotes = "Integridade do Sinal"
    msg_espera = "A aguardar ligação do recetor..."
    titulo_out = "🎵 Áudio Reconstruído"
    btn_atualizar = "🔄 Procurar pacotes na rede"
    msg_sem_dados = "A escutar a rede... Nenhum pacote recebido ainda."
    msg_dados_detetados = "Pacote de áudio detetado!"
else:
    titulo = "📡 Receiver - Team 10"
    btn_conectar = "🔌 Connect to Transmitter"
    msg_conectar_sucesso = "Signal established securely. Listening..."
    btn_reconstruir = "🧠 Start GenAI Reconstruction"
    msg_recon_sucesso = "Audio successfully reconstructed!"
    lbl_latencia = "Network Latency"
    lbl_pacotes = "Signal Integrity"
    msg_espera = "Waiting for receiver connection..."
    titulo_out = "🎵 Reconstructed Audio"
    btn_atualizar = "🔄 Check for network packages"
    msg_sem_dados = "Listening to network... No packages received yet."
    msg_dados_detetados = "Audio package detected!"

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
        st.success(msg_conectar_sucesso)

# Lógica de escuta da pasta output (Verifica ficheiros reais)
ficheiro_recente = None
if st.session_state.conectado:
    if os.path.exists(OUTPUT_DIR):
        ficheiros_wav = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]
        if ficheiros_wav:
            # Encontra o ficheiro mais recente gerado pela GenAI
            ficheiro_recente = max(ficheiros_wav, key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)))
            st.session_state.dados_prontos = True
        else:
            st.session_state.dados_prontos = False

with col2:
    st.subheader("2. Telemetria" if st.session_state.idioma == "Português 🇵🇹" else "2. Telemetry")
    if st.session_state.conectado:
        metrica_col1, metrica_col2 = st.columns(2)
        metrica_col1.metric(label=lbl_latencia, value=f"{st.session_state.latencia} ms", delta="-2 ms", delta_color="inverse")
        metrica_col2.metric(label=lbl_pacotes, value="100%", delta="OK")
        
        st.write("")
        # Se estiver conectado mas ainda não houver ficheiros, mostra botão para atualizar
        if not st.session_state.dados_prontos:
            st.info(msg_sem_dados)
            if st.button(btn_atualizar, use_container_width=True):
                st.rerun()
        else:
            st.success(f"{msg_dados_detetados} ({ficheiro_recente})")
    else:
        st.info(msg_espera)

st.write("")
st.write("")

if st.session_state.dados_prontos and ficheiro_recente:
    if st.button(btn_reconstruir, type="primary", use_container_width=True):
        barra_progresso = st.progress(0)
        # Mantemos o efeito visual da barra de progresso para a demo
        for percentagem in range(100):
            time.sleep(0.03)
            barra_progresso.progress(percentagem + 1)
            
        st.success(msg_recon_sucesso)
        
        st.divider()
        st.subheader(titulo_out)
        
        # Carrega o áudio real processado pela GenAI
        caminho_audio = os.path.join(OUTPUT_DIR, ficheiro_recente)
        st.audio(caminho_audio, format="audio/wav")