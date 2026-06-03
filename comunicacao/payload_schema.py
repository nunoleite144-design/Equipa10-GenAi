# =============================================================================
# payload_schema.py — Validação e descodificação do payload JSON da Equipa 9
# Equipa 10 — ESIST 2025/2026 — Módulo de Comunicação
# =============================================================================
#
# FORMATO REAL DA EQUIPA 9 (protocol_version 2.0):
#
#   {
#     "message_id": "uuid",
#     "timestamp":  1779906511,          ← Unix timestamp (int)
#     "topic":      "team9/messages",
#     "packet": {
#       "packet_id":          "uuid",
#       "packet_type":        "semantic_audio_codec",
#       "protocol_version":   "2.0",
#       "created_at_utc":     "ISO8601",
#       "sender":             { "team": "Transmitter", ... },
#       "language_hint":      "auto",
#       "audio_profile":      { "sample_rate_hz": 16000, ... },
#       "semantic_encoding":  { "token_count": 1952, ... },
#       "secure_payload": {
#         "cipher":         "AES-256-GCM",
#         "ciphertext_b64": "base64...",   ← ciphertext + GCM tag
#         "nonce_b64":      "base64..."    ← nonce de 12 bytes
#       },
#       "stt_profile":    {...},
#       "checksum_sha256": "hex"
#     }
#   }
#
#   Após desencriptar secure_payload obtemos:
#   {
#     "token_file":        "base64 do ficheiro .pt com os tokens SemanticCodec",
#     "transcript_text":   "transcrição Whisper",
#     "language_detected": "pt",
#     "stt_meta":          {...}
#   }
#
# COMPATIBILIDADE: também aceita os formatos de teste do publisher.py
#   (ciphertext, tokens_encrypted, tokens em claro)
#
# =============================================================================

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import torch

import config
from crypto import decrypt_team9, decrypt_tokens, decrypt_payload, key_from_hex, CryptoError

logger = logging.getLogger(__name__)


@dataclass
class SemanticPayload:
    """Representação interna de uma mensagem semântica recebida da Equipa 9."""
    message_id:        str
    timestamp:         str
    tokens:            Optional[list[int]] = None   # tokens SemanticCodec em claro
    text:              Optional[str] = None          # transcrição Whisper
    sample_rate:       int = 16000
    language:          Optional[str] = None
    speaker_label:     Optional[str] = None
    stt_meta:          dict = field(default_factory=dict)
    audio_profile:     dict = field(default_factory=dict)
    received_at:       str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def has_tokens(self) -> bool:
        return self.tokens is not None and len(self.tokens) > 0

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())


class PayloadValidationError(Exception):
    pass


def _load_tokens_from_pt_bytes(pt_bytes: bytes) -> list[int]:
    """Carrega tokens de bytes de um ficheiro .pt em memória."""
    buf = io.BytesIO(pt_bytes)
    data = torch.load(buf, weights_only=False)
    if isinstance(data, torch.Tensor):
        return [int(t) for t in data.flatten().tolist()]
    elif isinstance(data, dict) and "tokens" in data:
        t = data["tokens"]
        if isinstance(t, torch.Tensor):
            return [int(x) for x in t.flatten().tolist()]
        return [int(x) for x in t]
    raise PayloadValidationError("Ficheiro .pt não contém tensor de tokens reconhecível.")


def decode_and_validate(raw_payload: bytes) -> SemanticPayload:
    """
    Pipeline completo: bytes MQTT → desencriptar → validar → SemanticPayload.

    Suporta:
      1. Formato real Equipa 9 (protocol_version 2.0) com secure_payload
      2. Formato de teste publisher.py (ciphertext, tokens_encrypted, tokens)

    Raises:
        PayloadValidationError
    """
    # 1. Bytes → UTF-8
    try:
        text = raw_payload.decode("utf-8")
    except UnicodeDecodeError as e:
        raise PayloadValidationError(f"Payload não é UTF-8 válido: {e}") from e

    # 2. JSON → dict
    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as e:
        raise PayloadValidationError(f"Payload não é JSON válido: {e}") from e

    if not isinstance(data, dict):
        raise PayloadValidationError("Payload deve ser um objecto JSON.")

    key = key_from_hex(config.ENCRYPTION_KEY)

    # =========================================================================
    # FORMATO REAL EQUIPA 9 — detectado pela presença de "packet"
    # =========================================================================
    if "packet" in data and isinstance(data["packet"], dict):
        return _decode_team9_packet(data, key)

    # =========================================================================
    # FORMATOS DE TESTE (publisher.py) — retrocompatibilidade
    # =========================================================================
    return _decode_test_packet(data, key)


# -----------------------------------------------------------------------------
# Decoder — formato real Equipa 9
# -----------------------------------------------------------------------------

def _decode_team9_packet(data: dict, key: bytes) -> SemanticPayload:
    message_id = data.get("message_id", "unknown")
    timestamp  = str(data.get("timestamp", ""))
    packet     = data["packet"]

    secure = packet.get("secure_payload", {})
    ciphertext_b64 = secure.get("ciphertext_b64")
    nonce_b64      = secure.get("nonce_b64")

    if not ciphertext_b64 or not nonce_b64:
        raise PayloadValidationError("secure_payload incompleto — falta ciphertext_b64 ou nonce_b64.")

    # Desencriptar
    try:
        plaintext_bytes = decrypt_team9(ciphertext_b64, nonce_b64, key)
    except CryptoError as e:
        raise PayloadValidationError(f"Falha na desencriptação: {e}") from e

    try:
        inner = json.loads(plaintext_bytes.decode("utf-8"))
    except Exception as e:
        raise PayloadValidationError(f"Conteúdo desencriptado não é JSON válido: {e}") from e

    # Extrair ficheiro .pt em base64 e converter para lista de tokens
    token_file_b64 = inner.get("token_file")
    tokens = None
    if token_file_b64:
        try:
            pt_bytes = base64.b64decode(token_file_b64)
            tokens   = _load_tokens_from_pt_bytes(pt_bytes)
            logger.info("[Schema] ✓ Token file desencriptado e carregado: %d tokens", len(tokens))
        except Exception as e:
            raise PayloadValidationError(f"Falha ao carregar token_file: {e}") from e

    transcript = inner.get("transcript_text", "")
    language   = inner.get("language_detected")
    stt_meta   = inner.get("stt_meta", {})

    audio_profile = packet.get("audio_profile") or {}
    sample_rate   = int(audio_profile.get("sample_rate_hz", 16000))
    speaker_label = packet.get("sender", {}).get("speaker_label")

    if not tokens and not transcript:
        raise PayloadValidationError("Pacote não contém tokens nem transcrição.")

    if transcript:
        logger.info("[Schema] ✓ Transcrição: %s", transcript[:80])

    return SemanticPayload(
        message_id    = message_id,
        timestamp     = timestamp,
        tokens        = tokens,
        text          = transcript if transcript else None,
        sample_rate   = sample_rate,
        language      = language,
        speaker_label = speaker_label,
        stt_meta      = stt_meta,
        audio_profile = audio_profile,
    )


# -----------------------------------------------------------------------------
# Decoder — formatos de teste (publisher.py)
# -----------------------------------------------------------------------------

def _decode_test_packet(data: dict, key: bytes) -> SemanticPayload:
    for required in ("message_id", "timestamp"):
        if not data.get(required):
            raise PayloadValidationError(f"Campo obrigatório em falta: '{required}'.")

    tokens   = None
    text_dec = None

    if "ciphertext" in data and data["ciphertext"]:
        try:
            decrypted = decrypt_payload(data["ciphertext"], key)
            tokens    = decrypted.get("tokens")
            text_dec  = decrypted.get("text")
            logger.info("[Schema] ✓ ciphertext desencriptado: %d tokens", len(tokens) if tokens else 0)
        except CryptoError as e:
            raise PayloadValidationError(f"Falha na desencriptação (ciphertext): {e}") from e

    elif "tokens_encrypted" in data and data["tokens_encrypted"]:
        try:
            tokens   = decrypt_tokens(data["tokens_encrypted"], key)
            text_dec = data.get("text")
            logger.info("[Schema] ✓ tokens_encrypted desencriptado: %d tokens", len(tokens))
        except CryptoError as e:
            raise PayloadValidationError(f"Falha na desencriptação (tokens_encrypted): {e}") from e

    elif "tokens" in data and isinstance(data["tokens"], list):
        tokens   = data["tokens"]
        text_dec = data.get("text")
        logger.warning("[Schema] Tokens em CLARO — só para desenvolvimento!")

    if not tokens and not text_dec:
        raise PayloadValidationError("Payload não contém tokens nem texto.")

    sample_rate = int(data.get("sample_rate", 16000))

    return SemanticPayload(
        message_id = str(data["message_id"]).strip(),
        timestamp  = str(data["timestamp"]).strip(),
        tokens     = tokens,
        text       = text_dec.strip() if isinstance(text_dec, str) and text_dec.strip() else None,
        sample_rate= sample_rate,
        language   = data.get("language"),
    )
