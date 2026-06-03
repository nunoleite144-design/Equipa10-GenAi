# =============================================================================
# mqtt_receiver.py — Subscriber MQTT (Equipa 10)
# Equipa 10 — ESIST 2025/2026 — Módulo de Comunicação
# =============================================================================

import json
import logging
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

import config
from payload_schema import decode_and_validate, PayloadValidationError

logger = logging.getLogger("mqtt_receiver")


class MQTTReceiver:
    """
    Liga ao broker Mosquitto, subscreve o tópico da Equipa 9
    e chama on_message_cb(payload) para cada mensagem válida.
    """

    def __init__(self, on_message_cb):
        """
        on_message_cb: função chamada com um SemanticPayload já desencriptado.
        """
        self._cb = on_message_cb
        self._reconnect_count = 0

        self._client = mqtt.Client(client_id=config.CLIENT_ID)
        if config.BROKER_USERNAME:
            self._client.username_pw_set(config.BROKER_USERNAME, config.BROKER_PASSWORD)

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def start(self):
        logger.info("[MQTT] A ligar a %s:%d …", config.BROKER_HOST, config.BROKER_PORT)
        attempt = 0
        while True:
            try:
                self._client.connect(config.BROKER_HOST, config.BROKER_PORT, config.KEEPALIVE_SECONDS)
                self._client.loop_start()
                return
            except (ConnectionRefusedError, OSError) as e:
                attempt += 1
                logger.warning("[MQTT] Tentativa %d falhou: %s. A tentar em %ds…",
                               attempt, e, config.RECONNECT_DELAY_SECONDS)
                if attempt >= config.MAX_RECONNECT_ATTEMPTS:
                    raise ConnectionError("Não foi possível ligar ao broker.") from e
                time.sleep(config.RECONNECT_DELAY_SECONDS)

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("[MQTT] Desligado.")

    # ------------------------------------------------------------------
    # Callbacks internas
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._reconnect_count = 0
            client.subscribe(config.TOPIC_SUBSCRIBE, qos=config.QOS)
            logger.info("[MQTT] Ligado. A subscrever '%s'", config.TOPIC_SUBSCRIBE)
        else:
            logger.error("[MQTT] Falha na ligação (rc=%d)", rc)

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning("[MQTT] Desligado inesperadamente (rc=%d). A reconectar…", rc)
            self._reconnect_count += 1
            if self._reconnect_count <= config.MAX_RECONNECT_ATTEMPTS:
                time.sleep(config.RECONNECT_DELAY_SECONDS)
                try:
                    client.reconnect()
                except Exception as e:
                    logger.error("[MQTT] Erro na reconexão: %s", e)

    def _on_message(self, client, userdata, msg):
        logger.info("[MQTT] Mensagem recebida no tópico '%s'", msg.topic)
        try:
            payload = decode_and_validate(msg.payload)
            self._publish_ack(payload.message_id, "ok")
            self._cb(payload)
        except PayloadValidationError as e:
            logger.error("[MQTT] Payload inválido: %s", e)
            self._publish_ack("unknown", "error", str(e))
        except Exception as e:
            logger.exception("[MQTT] Erro inesperado: %s", e)

    def _publish_ack(self, message_id: str, status: str, detail: str = ""):
        ack = {
            "message_id": message_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if detail:
            ack["detail"] = detail
        self._client.publish(config.TOPIC_ACK, json.dumps(ack), qos=config.QOS)
