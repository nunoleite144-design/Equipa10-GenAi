"""Interface Streamlit para o Recetor SemantiCodec — Equipa 10.

Dois modos:
  • Modo Técnico — monitorização completa do watcher, upload manual, historial
  • Modo Demo    — interface simples e apelativa para demonstrações públicas
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
RECEIVED_DIR = BASE_DIR.parent / "comunicacao" / "received"

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Recetor GenAI — Equipa 10",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS global
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Modo Demo — cartões grandes */
.demo-step {
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 8px;
}
.demo-step-inactive { background: #1e2130; border: 2px solid #2d3250; }
.demo-step-active   { background: #1a3a5c; border: 2px dashed #4da6ff;
                       animation: pulse 1.5s infinite; }
.demo-step-done     { background: #12352a; border: 2px solid #2ecc71; }

@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(77,166,255,0.4); }
    70%  { box-shadow: 0 0 0 14px rgba(77,166,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(77,166,255,0); }
}

.demo-icon   { font-size: 56px; margin-bottom: 12px; }
.demo-title  { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
.demo-sub    { font-size: 14px; color: #aaa; }

.arrow-demo  { font-size: 36px; text-align: center; color: #4da6ff;
               padding: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------
defaults: dict = {
    "modo_demo": False,
    "auto_refresh": False,
    "last_refresh": 0.0,
    "watcher_pid": None,
    "manual_decode_result": None,
    "ddim_steps": 50,
    "gain": 1.5,
    "cfg_scale": 2.0,
    "device": "cpu",
    "demo_step": 0,
    "demo_audio": None,
    "demo_latencia": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ler_status_files() -> list[dict]:
    if not OUTPUT_DIR.exists():
        return []
    ficheiros = sorted(
        OUTPUT_DIR.glob("*_status.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    resultados = []
    for f in ficheiros:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_mtime"] = f.stat().st_mtime
            resultados.append(data)
        except Exception:
            pass
    return resultados


def ultimo_audio_concluido() -> tuple[str | None, dict | None]:
    for s in ler_status_files():
        if s.get("status") == "completed":
            p = s.get("audio_file")
            if p and Path(p).exists():
                return p, s
    return None, None


def cor_status(status: str) -> str:
    return {"completed": "🟢", "decoding": "🟡", "failed": "🔴"}.get(status, "⚪")


def _processo_vivo(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=3,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def watcher_ativo() -> bool:
    pid = st.session_state.watcher_pid
    if pid is None:
        return False
    if _processo_vivo(pid):
        return True
    st.session_state.watcher_pid = None
    return False


def iniciar_watcher(watch_dir: str) -> None:
    proc = subprocess.Popen(
        [
            sys.executable,
            str(BASE_DIR / "received_watcher.py"),
            "--watch-dir", watch_dir,
            "--output-dir", str(OUTPUT_DIR),
            "--ddim-steps", str(st.session_state.ddim_steps),
            "--cfg-scale", str(st.session_state.cfg_scale),
            "--gain", str(st.session_state.gain),
            "--device", st.session_state.device,
        ],
        cwd=str(BASE_DIR),
    )
    st.session_state.watcher_pid = proc.pid


def parar_watcher() -> None:
    pid = st.session_state.watcher_pid
    if pid:
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5)
        except Exception:
            pass
        st.session_state.watcher_pid = None


def formatar_latencia(ms: int | None) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms} ms"
    return f"{ms / 1000:.1f} s"


def formatar_shape(shape: list | None) -> str:
    if not shape:
        return "—"
    return " × ".join(str(d) for d in shape)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📡 Recetor GenAI")
    st.caption("Equipa 10 — SemantiCodec Decoder")
    st.divider()

    st.session_state.modo_demo = st.toggle(
        "🎭 Modo Demonstração",
        value=st.session_state.modo_demo,
        help="Ativa interface simplificada para demonstrações públicas",
    )

    st.divider()

    st.subheader("⚙️ Decoder")
    st.session_state.ddim_steps = st.slider("DDIM Steps", 10, 100, st.session_state.ddim_steps, 5)
    st.session_state.gain = st.slider("Ganho", 0.5, 3.0, st.session_state.gain, 0.1)
    st.session_state.cfg_scale = st.slider("CFG Scale", 0.0, 5.0, st.session_state.cfg_scale, 0.5)
    st.session_state.device = st.selectbox(
        "Dispositivo", ["cpu", "cuda", "mps", "auto"],
        index=["cpu", "cuda", "mps", "auto"].index(st.session_state.device),
    )

    st.divider()
    st.subheader("📂 Pasta Vigiada")
    watch_dir_input = st.text_input("Caminho", value=str(RECEIVED_DIR))

    if not st.session_state.modo_demo:
        st.divider()
        st.subheader("🔄 Auto-Refresh")
        st.session_state.auto_refresh = st.toggle("Ativo", value=st.session_state.auto_refresh)


# ===========================================================================
# MODO DEMO
# ===========================================================================
if st.session_state.modo_demo:

    st.markdown("""
    <div style="text-align:center; padding: 8px 0 24px 0;">
        <div style="font-size:42px;">📡</div>
        <div style="font-size:32px; font-weight:800; letter-spacing:1px;">
            Comunicação Semântica de Áudio
        </div>
        <div style="font-size:16px; color:#aaa; margin-top:6px;">
            A tua voz é transformada em código, enviada pela rede e reconstruída por Inteligência Artificial
        </div>
    </div>
    """, unsafe_allow_html=True)

    ativo = watcher_ativo()

    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        if not ativo:
            if st.button("▶  Ligar o Recetor", type="primary", use_container_width=True):
                iniciar_watcher(watch_dir_input)
                time.sleep(0.4)
                st.session_state.demo_step = 1
                st.session_state.demo_audio = None
                st.rerun()
        else:
            if st.button("⏹  Desligar o Recetor", use_container_width=True):
                parar_watcher()
                st.session_state.demo_step = 0
                st.rerun()

    st.write("")

    audio_path, audio_status = ultimo_audio_concluido()

    if audio_path and audio_path != st.session_state.demo_audio:
        st.session_state.demo_audio = audio_path
        st.session_state.demo_latencia = audio_status.get("decode_latency_ms") if audio_status else None
        st.session_state.demo_step = 3

    if ativo and st.session_state.demo_step < 3:
        decoding_now = any(s.get("status") == "decoding" for s in ler_status_files())
        st.session_state.demo_step = 2 if decoding_now else 1

    c1, ca, c2, cb, c3 = st.columns([3, 1, 3, 1, 3])
    step = st.session_state.demo_step

    def card_class(n: int) -> str:
        if step == n:
            return "demo-step demo-step-active"
        if step > n:
            return "demo-step demo-step-done"
        return "demo-step demo-step-inactive"

    def check(n: int) -> str:
        return "✅" if step > n else ""

    with c1:
        st.markdown(f"""
        <div class="{card_class(1)}">
            <div class="demo-icon">🎙️</div>
            <div class="demo-title">1. Voz Recebida {check(1)}</div>
            <div class="demo-sub">A aguardar sinal da rede...</div>
        </div>
        """, unsafe_allow_html=True)

    with ca:
        st.markdown('<div class="arrow-demo" style="margin-top:60px;">→</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="{card_class(2)}">
            <div class="demo-icon">🧠</div>
            <div class="demo-title">2. IA a Reconstruir {check(2)}</div>
            <div class="demo-sub">Inteligência Artificial a regenerar a voz...</div>
        </div>
        """, unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="arrow-demo" style="margin-top:60px;">→</div>', unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="{card_class(3)}">
            <div class="demo-icon">🔊</div>
            <div class="demo-title">3. Áudio Pronto {check(3)}</div>
            <div class="demo-sub">Pronto para ouvir!</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if step == 3 and st.session_state.demo_audio:
        st.markdown("""
        <div style="text-align:center; font-size:22px; font-weight:700;
                    color:#2ecc71; margin: 16px 0 8px 0;">
            ✅ Voz reconstruída com sucesso!
        </div>
        """, unsafe_allow_html=True)

        audio_col = st.columns([1, 3, 1])[1]
        with audio_col:
            st.audio(st.session_state.demo_audio, format="audio/wav")

        if st.session_state.demo_latencia:
            lat_s = st.session_state.demo_latencia / 1000
            st.markdown(f"""
            <div style="text-align:center; color:#aaa; font-size:13px; margin-top:4px;">
                A Inteligência Artificial demorou <b style="color:#fff">{lat_s:.1f} segundos</b>
                a reconstruir esta voz a partir de código semântico
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        reset_col = st.columns([1, 2, 1])[1]
        with reset_col:
            if st.button("🔁  Aguardar próxima mensagem", use_container_width=True):
                st.session_state.demo_step = 1
                st.session_state.demo_audio = None
                st.rerun()

    elif step == 2:
        prog_col = st.columns([1, 3, 1])[1]
        with prog_col:
            st.progress(0.6, text="🧠 A reconstruir áudio com IA...")

    elif step == 1 and ativo:
        info_col = st.columns([1, 3, 1])[1]
        with info_col:
            st.info("📶 Recetor ligado. À espera de mensagem do transmissor...")

    elif step == 0:
        info_col = st.columns([1, 3, 1])[1]
        with info_col:
            st.warning("🔌 Liga o recetor para começar.")

    st.write("")
    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#888; font-size:13px; max-width:600px; margin:0 auto;">
        <b style="color:#aaa">Como funciona?</b><br>
        Em vez de enviar o áudio completo pela rede (que ocupa muito espaço),
        o transmissor extrai apenas o <i>significado</i> da voz — tokens semânticos.
        No recetor, uma Inteligência Artificial reconstrói o áudio a partir desses tokens,
        como se "imaginasse" de volta a voz original.
    </div>
    """, unsafe_allow_html=True)

    if ativo and step < 3:
        time.sleep(2.5)
        st.rerun()

    st.stop()


# ===========================================================================
# MODO TÉCNICO
# ===========================================================================
st.title("📡 Recetor Semântico de Áudio")
st.caption("Equipa 10 · SemantiCodec + GenAI · Receção e Reconstrução de Áudio")
st.divider()

# Secção 1 — Watcher
col_w1, col_w2, col_w3 = st.columns([2, 1, 1])
with col_w1:
    st.subheader("1. Serviço de Escuta (Watcher)")

ativo = watcher_ativo()

with col_w2:
    if ativo:
        st.metric("Estado", "🟢 Ativo", delta=f"PID {st.session_state.watcher_pid}")
    else:
        st.metric("Estado", "🔴 Inativo")

with col_w3:
    if ativo:
        if st.button("⏹ Parar Watcher", use_container_width=True):
            parar_watcher()
            st.rerun()
    else:
        if st.button("▶ Iniciar Watcher", type="primary", use_container_width=True):
            iniciar_watcher(watch_dir_input)
            time.sleep(0.5)
            st.rerun()

if ativo:
    st.info(
        f"A vigiar **{watch_dir_input}** → **{OUTPUT_DIR}**  |  "
        f"DDIM {st.session_state.ddim_steps} passos · Gain {st.session_state.gain} · "
        f"CFG {st.session_state.cfg_scale} · {st.session_state.device.upper()}"
    )
else:
    st.warning("O Watcher não está ativo. Inicia-o para processar mensagens recebidas automaticamente.")

st.divider()

# Secção 2 — Upload manual
st.subheader("2. Descodificação Manual (Upload de Payload)")

with st.expander("Carregar ficheiro JSON de payload", expanded=False):
    ficheiro_upload = st.file_uploader(
        "Arrastra ou seleciona um payload JSON (formato Comunicação ou canónico SemantiCodec)",
        type=["json"],
        key="payload_uploader",
    )

    col_m1, col_m2 = st.columns(2)
    ddim_manual = col_m1.number_input("DDIM Steps", min_value=10, max_value=100, value=st.session_state.ddim_steps, step=5)
    gain_manual = col_m2.number_input("Gain", min_value=0.1, max_value=5.0, value=st.session_state.gain, step=0.1)

    if ficheiro_upload and st.button("🧠 Descodificar Payload", type="primary"):
        payload_bytes = ficheiro_upload.read()
        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError as e:
            st.error(f"JSON inválido: {e}")
            payload = None

        if payload is not None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with st.spinner("A carregar modelo e a reconstruir áudio... (pode demorar 1-2 min na primeira vez)"):
                sys.path.insert(0, str(BASE_DIR))
                try:
                    from comunicacao_adapter import decode_comm_payload, resolve_params
                    from semantic_receiver import load_model

                    token_rate, vocab, sample_rate = resolve_params(payload)
                    model = load_model(
                        token_rate=token_rate,
                        semantic_vocab_size=vocab,
                        device=st.session_state.device,
                        ddim_steps=int(ddim_manual),
                        cfg_scale=st.session_state.cfg_scale,
                    )
                    result = decode_comm_payload(
                        payload,
                        token_rate=token_rate,
                        semantic_vocab_size=vocab,
                        sample_rate=sample_rate,
                        output_dir=OUTPUT_DIR,
                        gain=float(gain_manual),
                        device=st.session_state.device,
                        ddim_steps=int(ddim_manual),
                        cfg_scale=st.session_state.cfg_scale,
                        model=model,
                    )
                    st.session_state.manual_decode_result = {
                        "message_id": result.message_id,
                        "audio_file": str(result.audio_file),
                        "decode_latency_ms": result.decode_latency_ms,
                        "tokens_shape": result.tokens_shape,
                        "token_rate": result.token_rate,
                        "semantic_vocab_size": result.semantic_vocab_size,
                    }
                    st.success(f"Áudio reconstruído em {formatar_latencia(result.decode_latency_ms)}!")
                except Exception as exc:
                    st.error(f"Erro ao descodificar: {exc}")

    res = st.session_state.manual_decode_result
    if res:
        audio_path = Path(res["audio_file"])
        if audio_path.exists():
            st.audio(str(audio_path), format="audio/wav")
            c1, c2, c3 = st.columns(3)
            c1.metric("Latência Decode", formatar_latencia(res.get("decode_latency_ms")))
            c2.metric("Tokens Shape", formatar_shape(res.get("tokens_shape")))
            c3.metric("Token Rate", f"{res.get('token_rate', '—')} Hz")

st.divider()

# Secção 3 — Histórico
st.subheader("3. Mensagens Recebidas")

col_r1, col_r2 = st.columns([4, 1])
with col_r2:
    if st.button("🔄 Atualizar", use_container_width=True):
        st.rerun()

status_list = ler_status_files()

if not status_list:
    st.info("Nenhuma mensagem processada ainda. Inicia o Watcher e aguarda pacotes da Comunicação.")
else:
    total = len(status_list)
    concluidos = sum(1 for s in status_list if s.get("status") == "completed")
    em_decode = sum(1 for s in status_list if s.get("status") == "decoding")
    falhados = sum(1 for s in status_list if s.get("status") == "failed")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", total)
    m2.metric("Concluídos", concluidos)
    m3.metric("Em Decode", em_decode)
    m4.metric("Falhados", falhados)

    st.write("")

    for entrada in status_list:
        status = entrada.get("status", "unknown")
        msg_id = entrada.get("message_id", "—")
        icone = cor_status(status)
        latencia = entrada.get("decode_latency_ms")
        audio_file = entrada.get("audio_file")
        tokens_shape = entrada.get("tokens_shape")
        token_rate = entrada.get("token_rate")
        vocab = entrada.get("semantic_vocab_size")
        ddim = entrada.get("ddim_steps")
        gain_val = entrada.get("gain")
        erro = entrada.get("error_message")
        mtime = entrada.get("_mtime", 0)
        ts = time.strftime("%H:%M:%S", time.localtime(mtime)) if mtime else "—"

        label = f"{icone} `{msg_id}` — {status.upper()} — {ts}"
        with st.expander(label, expanded=(status == "decoding")):
            if status == "completed" and audio_file:
                audio_p = Path(audio_file)
                if audio_p.exists():
                    st.audio(str(audio_p), format="audio/wav")
                    ca, cb, cc, cd = st.columns(4)
                    ca.metric("Latência Decode", formatar_latencia(latencia))
                    cb.metric("Tokens", formatar_shape(tokens_shape))
                    cc.metric("Token Rate", f"{token_rate} Hz" if token_rate else "—")
                    cd.metric("DDIM Steps", ddim or "—")
                    ce, cf = st.columns(2)
                    ce.metric("Vocab Size", vocab or "—")
                    cf.metric("Gain", gain_val or "—")
                else:
                    st.warning(f"Ficheiro não encontrado: {audio_file}")
            elif status == "decoding":
                st.info("A descodificar...")
                st.progress(0.5)
            elif status == "failed":
                st.error(f"**Fase:** {entrada.get('stage','')} | **Código:** {entrada.get('error_code','')}")
                if erro:
                    st.code(erro, language="text")
            else:
                st.json({k: v for k, v in entrada.items() if not k.startswith("_")})

st.divider()

# Auto-refresh (modo técnico)
if st.session_state.auto_refresh:
    now = time.time()
    elapsed = now - st.session_state.last_refresh
    if elapsed >= 3.0:
        st.session_state.last_refresh = now
        st.rerun()
    else:
        time.sleep(3.0 - elapsed)
        st.session_state.last_refresh = time.time()
        st.rerun()

st.caption("Equipa 10 · GenAI Receiver · SemantiCodec Decoder")
