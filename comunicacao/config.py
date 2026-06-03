# =============================================================================
# config.py — Configurações centrais do módulo de comunicação MQTT
# Equipa 10 — ESIST 2025/2026 — Módulo de Comunicação
# =============================================================================

# -----------------------------------------------------------------------------
# Broker Mosquitto
# -----------------------------------------------------------------------------
BROKER_HOST = "localhost"
BROKER_PORT = 1883
BROKER_USERNAME = None
BROKER_PASSWORD = None
CLIENT_ID = "equipa10_receiver"

# -----------------------------------------------------------------------------
# Tópicos MQTT
# -----------------------------------------------------------------------------
TOPIC_SUBSCRIBE = "team9/messages"      # tópico real da Equipa 9
TOPIC_ACK       = "semantic/tx"         # tópico de feedback deles
QOS = 1

# -----------------------------------------------------------------------------
# Ligação e Reconexão
# -----------------------------------------------------------------------------
KEEPALIVE_SECONDS       = 60
RECONNECT_DELAY_SECONDS = 5
MAX_RECONNECT_ATTEMPTS  = 10

# -----------------------------------------------------------------------------
# Pipeline interno
# -----------------------------------------------------------------------------
QUEUE_MAX_SIZE = 100
LOG_LEVEL      = "INFO"

# -----------------------------------------------------------------------------
# Encriptação AES-256-GCM
# -----------------------------------------------------------------------------
# Chave partilhada com a Equipa 9.
# Gerada com: python -c "from crypto import generate_key; print(generate_key())"
#
# IMPORTANTE:
#   - Deve ter exactamente 64 caracteres hexadecimais (= 32 bytes = 256 bits)
#   - A Equipa 9 tem de usar exactamente a mesma string aqui
#   - Nunca colocar esta chave num repositório público (git, etc.)
#
ENCRYPTION_KEY = "e0788803ee19a0609500bca03dc7541ad25c198419c5eebc729a1bff569aab7b"
