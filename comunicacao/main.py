# =============================================================================
# main.py — Receiver MQTT (Equipa 10)
# Equipa 10 — ESIST 2025/2026 — Módulo de Comunicação
#
# Uso:
#   python main.py
#
# Cada mensagem válida recebida é guardada em:
#   received/TIMESTAMP_MESSAGEID.json
#
# A equipa de AI carrega assim:
#   import json
#   with open("received/....json", encoding="utf-8") as f:
#       data = json.load(f)
#   tokens = data["tokens"]   # lista de inteiros SemanticCodec
#   text   = data["text"]     # string com transcrição Whisper (ou None)
# =============================================================================

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import config
from mqtt_receiver import MQTTReceiver
from payload_schema import SemanticPayload

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# -----------------------------------------------------------------------------
# Pasta de saída para a equipa de AI
# -----------------------------------------------------------------------------
OUTPUT_DIR = "received"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Processamento de cada mensagem recebida
# =============================================================================

def guardar_mensagem(payload: SemanticPayload) -> str:
    """
    Guarda o payload desencriptado como ficheiro JSON em received/.
    Retorna o caminho do ficheiro criado.

    A equipa de AI carrega com:
        import json
        with open("received/....json", encoding="utf-8") as f:
            data = json.load(f)
        tokens = data["tokens"]   # lista de inteiros
        text   = data["text"]     # str ou None
    """
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{timestamp_str}_{payload.message_id}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    data = {
        "tokens":      payload.tokens,
        "text":        payload.text,
        "sample_rate": payload.sample_rate,
        "message_id":  payload.message_id,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "timestamp":   payload.timestamp,
        "speaker_label": payload.speaker_label,
        "language":    payload.language,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return filepath


def on_message(payload: SemanticPayload) -> None:
    """
    Chamada pelo MQTTReceiver para cada mensagem válida.
    """
    logger.info("═" * 55)
    logger.info("[RX] Nova mensagem — id=%s", payload.message_id)

    if payload.has_tokens:
        logger.info("[RX] Tokens SemanticCodec : %d tokens  (sample_rate=%d Hz)",
                    len(payload.tokens), payload.sample_rate)
    if payload.has_text:
        logger.info("[RX] Transcrição Whisper  : %s", payload.text)

    filepath = guardar_mensagem(payload)
    logger.info("[RX] Guardado em → %s", filepath)
    logger.info("═" * 55)


# =============================================================================
# Arranque
# =============================================================================

def main():
    logger.info("═" * 55)
    logger.info("  ESIST 2025/2026 — Equipa 10 — Receiver MQTT")
    logger.info("  Broker : %s:%d", config.BROKER_HOST, config.BROKER_PORT)
    logger.info("  Tópico : %s", config.TOPIC_SUBSCRIBE)
    logger.info("  Saída  : ./%s/", OUTPUT_DIR)
    logger.info("═" * 55)

    receiver = MQTTReceiver(on_message_cb=on_message)

    def shutdown(sig, frame):
        logger.info("\n[Main] A encerrar…")
        receiver.stop()
        logger.info("[Main] Encerrado.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    receiver.start()
    logger.info("[Main] À escuta… (Ctrl+C para parar)")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()