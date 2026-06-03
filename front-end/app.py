"""Interface Streamlit para o Recetor SemantiCodec — Equipa 10.

Monitoriza a pasta ``output/`` gerada pelo received_watcher.py e permite
descodificar payloads manualmente via upload de ficheiro JSON.
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
# Estado da sessão
# ---------------------------------------------------------------------------
defaults = {
    "auto_refresh": False,
    "last_refresh": 0.0,
    "watcher_pid": None,
    "manual_decode_result": None,
    "ddim_steps": 50,
    "gain": 1.5,
    "cfg_scale": 2.0,
    "device": "cpu",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ler_status_files() -> list[dict]:
    """Lê todos os _status.json da pasta output/ e devolve lista ordenada (mais recente primeiro)."""
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


def cor_status(status: str) -> str:
    return {"completed": "🟢", "decoding": "🟡", "failed": "🔴"}.get(status, "⚪")


def _processo_vivo(pid: int) -> bool:
    """Verifica se um PID está ativo (compatível com Windows e Unix)."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    # Fallback: tenta via tasklist no Windows
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
# Sidebar — Configurações
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📡 Recetor GenAI")
    st.caption("Equipa 10 — SemantiCodec Decoder")
    st.divider()

    st.subheader("⚙️ Configurações do Decoder")

    st.session_state.ddim_steps = st.slider(
        "DDIM Steps", min_value=10, max_value=100, value=st.session_state.ddim_steps, step=5,
        help="Mais passos = maior qualidade, mais lento",
    )
    st.session_state.gain = st.slider(
        "Ganho de Áudio", min_value=0.5, max_value=3.0, value=st.session_state.gain, step=0.1,
        help="Amplificação após normalização",
    )
    st.session_state.cfg_scale = st.slider(
        "CFG Scale", min_value=0.0, max_value=5.0, value=st.session_state.cfg_scale, step=0.5,
        help="Escala de guidance do diffusion decoder",
    )
    st.session_state.device = st.selectbox(
        "Dispositivo", ["cpu", "cuda", "mps", "auto"],
        index=["cpu", "cuda", "mps", "auto"].index(st.session_state.device),
    )

    st.divider()

    st.subheader("📂 Pasta Vigiada")
    watch_dir_input = st.text_input(
        "Caminho",
        value=str(RECEIVED_DIR),
        help="Pasta onde a Comunicação deposita os JSON",
    )

    st.divider()
    st.subheader("🔄 Auto-Refresh")
    st.session_state.auto_refresh = st.toggle("Atualizar automaticamente", value=st.session_state.auto_refresh)
    if st.session_state.auto_refresh:
        st.caption("A interface atualiza a cada 3 segundos.")

# ---------------------------------------------------------------------------
# Cabeçalho principal
# ---------------------------------------------------------------------------
st.title("📡 Recetor Semântico de Áudio")
st.caption("Equipa 10 · SemantiCodec + GenAI · Receção e Reconstrução de Áudio")
st.divider()

# ---------------------------------------------------------------------------
# Secção 1 — Watcher (daemon de escuta)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Secção 2 — Upload manual de payload JSON
# ---------------------------------------------------------------------------
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
            # Gravar temporariamente em received/ para o decode usar o adapter
            tmp_path = OUTPUT_DIR / f"_manual_{ficheiro_upload.name}"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(payload_bytes)

            with st.spinner("A carregar modelo e a reconstruir áudio... (pode demorar 1-2 min na primeira vez)"):
                import sys as _sys
                _sys.path.insert(0, str(BASE_DIR))
                try:
                    from comunicacao_adapter import decode_comm_payload, read_comm_payload, resolve_params
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
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

    # Mostrar resultado do decode manual
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

# ---------------------------------------------------------------------------
# Secção 3 — Histórico de mensagens recebidas
# ---------------------------------------------------------------------------
st.subheader("3. Mensagens Recebidas")

col_r1, col_r2 = st.columns([4, 1])
with col_r2:
    if st.button("🔄 Atualizar", use_container_width=True):
        st.rerun()

status_list = ler_status_files()

if not status_list:
    st.info("Nenhuma mensagem processada ainda. Inicia o Watcher e aguarda pacotes da Comunicação.")
else:
    # Métricas resumo
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

    # Lista de mensagens
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
                audio_path = Path(audio_file)
                if audio_path.exists():
                    st.audio(str(audio_path), format="audio/wav")
                    ca, cb, cc, cd = st.columns(4)
                    ca.metric("Latência Decode", formatar_latencia(latencia))
                    cb.metric("Tokens", formatar_shape(tokens_shape))
                    cc.metric("Token Rate", f"{token_rate} Hz" if token_rate else "—")
                    cd.metric("DDIM Steps", ddim or "—")
                    ce, cf = st.columns(2)
                    ce.metric("Vocab Size", vocab or "—")
                    cf.metric("Gain", gain_val or "—")
                else:
                    st.warning(f"Ficheiro de áudio não encontrado: {audio_file}")

            elif status == "decoding":
                st.info("A descodificar... aguarda.")
                st.progress(0.5)

            elif status == "failed":
                stage = entrada.get("stage", "")
                err_code = entrada.get("error_code", "")
                st.error(f"**Fase:** {stage} | **Código:** {err_code}")
                if erro:
                    st.code(erro, language="text")

            else:
                st.json({k: v for k, v in entrada.items() if not k.startswith("_")})

st.divider()

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if st.session_state.auto_refresh:
    now = time.time()
    if now - st.session_state.last_refresh >= 3.0:
        st.session_state.last_refresh = now
        time.sleep(3.0)
        st.rerun()
    else:
        remaining = 3.0 - (now - st.session_state.last_refresh)
        time.sleep(max(0.1, remaining))
        st.session_state.last_refresh = time.time()
        st.rerun()

st.caption("Equipa 10 · GenAI Receiver · SemantiCodec Decoder")
