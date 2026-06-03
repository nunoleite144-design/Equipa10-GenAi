# =============================================================================
# crypto.py — Encriptação/Desencriptação AES-256-GCM dos tokens SemanticCodec
# Equipa 10 — ESIST 2025/2026 — Módulo de Comunicação
#
# PARTILHAR ESTE FICHEIRO COM A EQUIPA 9 — ambas as equipas têm de usar
# exactamente o mesmo código e a mesma chave (definida em config.py).
#
# Porquê AES-256-GCM?
#   - 256 bits de chave (o máximo do AES)
#   - GCM = modo autenticado: detecta automaticamente se os dados foram
#     adulterados no canal (garante confidencialidade + integridade)
#   - Cada mensagem tem um nonce aleatório de 12 bytes → mesmo que a
#     Equipa 9 envie a mesma frase duas vezes, os bytes encriptados
#     são sempre diferentes
#
# Formato do campo "tokens_encrypted" no payload JSON:
#   base64( nonce[12 bytes] + ciphertext + tag[16 bytes] )
#   Tudo junto numa string base64 — fácil de meter num campo JSON.
# =============================================================================

import os
import json
import base64
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# Tamanhos fixos (não alterar)
NONCE_SIZE = 12    # bytes — recomendado para GCM
KEY_SIZE   = 32    # bytes — AES-256


class CryptoError(Exception):
    """Levantada quando a desencriptação falha (chave errada, dados corrompidos, etc.)."""
    pass


def encrypt_tokens(tokens: list[int], key: bytes) -> str:
    """
    Encripta uma lista de tokens SemanticCodec com AES-256-GCM.

    Usado pela Equipa 9 antes de publicar no MQTT.

    Args:
        tokens: lista de inteiros do SemanticCodec.
        key:    chave AES de 32 bytes (de config.ENCRYPTION_KEY).

    Returns:
        String base64 com nonce + ciphertext + tag — pronta para meter no JSON.
    """
    if len(key) != KEY_SIZE:
        raise CryptoError(f"Chave inválida: deve ter {KEY_SIZE} bytes, tem {len(key)}.")

    # Serializar tokens para JSON bytes
    plaintext = json.dumps(tokens).encode("utf-8")

    # Gerar nonce aleatório (12 bytes) — único para cada mensagem
    nonce = os.urandom(NONCE_SIZE)

    # Encriptar
    aesgcm = AESGCM(key)
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    # Concatenar nonce + ciphertext+tag e codificar em base64
    result = base64.b64encode(nonce + ciphertext_with_tag).decode("ascii")

    logger.debug("[Crypto] Tokens encriptados: %d tokens → %d bytes base64",
                 len(tokens), len(result))
    return result


def decrypt_tokens(encrypted_b64: str, key: bytes) -> list[int]:
    """
    Desencripta o campo "tokens_encrypted" do payload e devolve a lista de tokens.

    Usado pelo receiver (Equipa 10) antes de passar à equipa de AI.

    Args:
        encrypted_b64: string base64 recebida no campo "tokens_encrypted".
        key:           chave AES de 32 bytes (de config.ENCRYPTION_KEY).

    Returns:
        Lista de inteiros (tokens SemanticCodec).

    Raises:
        CryptoError: se a chave for errada, os dados estiverem corrompidos
                     ou o payload tiver sido adulterado.
    """
    if len(key) != KEY_SIZE:
        raise CryptoError(f"Chave inválida: deve ter {KEY_SIZE} bytes, tem {len(key)}.")

    # Descodificar base64
    try:
        raw = base64.b64decode(encrypted_b64)
    except Exception as e:
        raise CryptoError(f"Campo 'tokens_encrypted' não é base64 válido: {e}") from e

    if len(raw) < NONCE_SIZE + 16:  # nonce + tag mínimo
        raise CryptoError("Dados encriptados demasiado curtos — corrompidos?")

    # Separar nonce do ciphertext+tag
    nonce      = raw[:NONCE_SIZE]
    ciphertext = raw[NONCE_SIZE:]

    # Desencriptar (GCM verifica a autenticidade automaticamente)
    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    except Exception as e:
        raise CryptoError(
            "Desencriptação falhou — chave errada ou dados adulterados no canal."
        ) from e

    # Deserializar JSON → lista de inteiros
    try:
        tokens = json.loads(plaintext.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise CryptoError(f"Tokens desencriptados não são JSON válido: {e}") from e

    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise CryptoError("Tokens desencriptados não são uma lista de inteiros.")

    logger.debug("[Crypto] Tokens desencriptados: %d tokens", len(tokens))
    return tokens


def decrypt_team9(ciphertext_b64: str, nonce_b64: str, key: bytes) -> bytes:
    """
    Desencripta o formato da Equipa 9, onde nonce e ciphertext vêm em campos separados.

    Usado em payload_schema.py para desencriptar secure_payload do pacote deles:
        {
            "cipher": "AES-256-GCM",
            "ciphertext_b64": "...",
            "nonce_b64": "..."
        }

    Returns:
        Bytes do plaintext (JSON com token_file, transcript_text, etc.)

    Raises:
        CryptoError: se a chave for errada ou os dados estiverem corrompidos.
    """
    if len(key) != KEY_SIZE:
        raise CryptoError(f"Chave inválida: deve ter {KEY_SIZE} bytes, tem {len(key)}.")

    try:
        nonce      = base64.b64decode(nonce_b64)
        ciphertext = base64.b64decode(ciphertext_b64)
    except Exception as e:
        raise CryptoError(f"nonce_b64 ou ciphertext_b64 não é base64 válido: {e}") from e

    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    except Exception as e:
        raise CryptoError("Desencriptação falhou — chave errada ou dados adulterados.") from e

    logger.debug("[Crypto] Team9 payload desencriptado: %d bytes", len(plaintext))
    return plaintext


def encrypt_payload(tokens: list[int], text: str | None, key: bytes) -> str:
    """
    Encripta tokens + texto juntos num único ciphertext AES-256-GCM.

    Usado pela Equipa 9 quando quer mandar tudo num único campo "ciphertext".

    Args:
        tokens: lista de inteiros do SemanticCodec.
        text:   transcrição Whisper (pode ser None).
        key:    chave AES de 32 bytes.

    Returns:
        String base64 com nonce + ciphertext + tag.
    """
    if len(key) != KEY_SIZE:
        raise CryptoError(f"Chave inválida: deve ter {KEY_SIZE} bytes, tem {len(key)}.")

    plaintext = json.dumps({"tokens": tokens, "text": text}, ensure_ascii=False).encode("utf-8")
    nonce = os.urandom(NONCE_SIZE)
    aesgcm = AESGCM(key)
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    result = base64.b64encode(nonce + ciphertext_with_tag).decode("ascii")

    logger.debug("[Crypto] Payload encriptado: %d tokens → %d bytes base64", len(tokens), len(result))
    return result


def decrypt_payload(encrypted_b64: str, key: bytes) -> dict:
    """
    Desencripta o campo "ciphertext" e devolve dict com tokens e texto.

    Returns:
        {"tokens": list[int], "text": str | None}

    Raises:
        CryptoError: se a chave for errada ou os dados estiverem corrompidos.
    """
    if len(key) != KEY_SIZE:
        raise CryptoError(f"Chave inválida: deve ter {KEY_SIZE} bytes, tem {len(key)}.")

    try:
        raw = base64.b64decode(encrypted_b64)
    except Exception as e:
        raise CryptoError(f"Campo 'ciphertext' não é base64 válido: {e}") from e

    if len(raw) < NONCE_SIZE + 16:
        raise CryptoError("Dados encriptados demasiado curtos — corrompidos?")

    nonce      = raw[:NONCE_SIZE]
    ciphertext = raw[NONCE_SIZE:]

    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    except Exception as e:
        raise CryptoError("Desencriptação falhou — chave errada ou dados adulterados no canal.") from e

    try:
        data = json.loads(plaintext.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise CryptoError(f"Payload desencriptado não é JSON válido: {e}") from e

    if "tokens" not in data or not isinstance(data["tokens"], list):
        raise CryptoError("Payload desencriptado não contém campo 'tokens' válido.")

    logger.debug("[Crypto] Payload desencriptado: %d tokens", len(data["tokens"]))
    return data


def generate_key() -> str:
    """
    Gera uma chave AES-256 aleatória e devolve-a como string hex.

    Usar uma vez para gerar a chave partilhada entre as duas equipas:
        python -c "from crypto import generate_key; print(generate_key())"

    Copiar o resultado para config.ENCRYPTION_KEY nas duas equipas.
    """
    return os.urandom(KEY_SIZE).hex()


def key_from_hex(hex_string: str) -> bytes:
    """
    Converte a chave hex do config.py para bytes usáveis pelo AES.

    Args:
        hex_string: string de 64 caracteres hex (ex: "a3f1...").

    Returns:
        32 bytes prontos para encrypt_tokens / decrypt_tokens.
    """
    try:
        key = bytes.fromhex(hex_string)
    except ValueError as e:
        raise CryptoError(f"ENCRYPTION_KEY inválida — deve ser 64 caracteres hex: {e}") from e

    if len(key) != KEY_SIZE:
        raise CryptoError(
            f"ENCRYPTION_KEY inválida — deve ter 64 caracteres hex ({KEY_SIZE} bytes), "
            f"tem {len(hex_string)} caracteres."
        )
    return key
