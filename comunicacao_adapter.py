#!/usr/bin/env python3
"""Adapter: Comunicação JSON -> SemantiCodec decode core.

The Comunicação module (Equipa 10) receives the message over MQTT, decrypts it
and hands the GenAI receiver a JSON file with the SemantiCodec tokens already in
plain form. This module bridges that wire format to the shared decode core in
``semantic_receiver`` WITHOUT touching the clean canonical contract in
``semantic_payload`` (which uses base64 torch tensors).

Comunicação wire format (plaintext tokens):

    {
      "message_id":          "<uuid>",                  # required
      "timestamp":           "<iso8601>",               # optional
      "tokens":              [[semantic, acoustic], ...] # required, shape [N, 2]
      "text":                "...",                      # optional Whisper, display only
      "sample_rate":         16000,                      # optional
      "token_rate":          100,                        # SHOULD be present
      "semantic_vocab_size": 16384,                      # SHOULD be present
      ...
    }

`token_rate` and `semantic_vocab_size` MUST match the SemantiCodec settings the
Equipa 9 transmitter used to encode. A mismatch produces unintelligible audio,
not a clean error, so when they are missing we fall back to defaults AND warn
loudly. The right fix is for Equipa 9 to include them in the JSON.

The `text` (Whisper) field is intentionally ignored: it is display-only metadata
for the WebApp and must never feed the audio reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from semantic_payload import SAMPLE_RATE
from semantic_receiver import DecodeResult, decode_tokens

# Fallbacks used only when the message omits these fields. Keep in sync with the
# values Equipa 9 encodes with.
DEFAULT_TOKEN_RATE = 100
DEFAULT_SEMANTIC_VOCAB_SIZE = 16384

VALID_TOKEN_RATES = (25, 50, 100)
VALID_VOCAB_SIZES = (4096, 8192, 16384, 32768)


class AdapterError(Exception):
    """User-facing error for malformed Comunicação payloads."""


def tokens_to_tensor(raw_tokens: Any) -> torch.Tensor:
    """Convert the Comunicação token list into a ``[1, N, 2]`` long tensor.

    Accepts ``[[s, a], ...]`` (shape ``[N, 2]``) or an already-batched
    ``[[[s, a], ...]]`` (shape ``[1, N, 2]``). SemantiCodec's ``decode`` expects
    ``[batch, length, 2]``.
    """
    if raw_tokens is None:
        raise AdapterError("Payload has no 'tokens'.")
    try:
        tokens = torch.tensor(raw_tokens, dtype=torch.long)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"Could not parse 'tokens' into a tensor: {exc}") from exc

    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)
    elif tokens.dim() != 3:
        raise AdapterError(
            f"Unexpected tokens shape {list(tokens.shape)}; expected [N, 2] or [1, N, 2]."
        )

    if tokens.numel() == 0:
        raise AdapterError("Payload 'tokens' is empty.")
    if tokens.shape[-1] != 2:
        raise AdapterError(
            f"SemantiCodec expects 2 columns (semantic, acoustic); "
            f"got {tokens.shape[-1]} in shape {list(tokens.shape)}."
        )
    return tokens


def _resolve_param(payload: dict, key: str, default: int, valid: tuple[int, ...]) -> int:
    value = payload.get(key)
    if value is None:
        print(
            f"[adapter] WARNING: '{key}' missing from payload; falling back to "
            f"{default}. This MUST match the Equipa 9 encode or the audio will be "
            f"garbage."
        )
        return default
    value = int(value)
    if value not in valid:
        raise AdapterError(f"'{key}' must be one of {valid}; got {value}.")
    return value


def resolve_params(payload: dict) -> tuple[int, int, int]:
    """Return ``(token_rate, semantic_vocab_size, sample_rate)`` for a payload.

    Reads them from the message, falling back to the module defaults (with a
    warning) when absent. Exposed so the watcher can resolve once, build the
    decoder once, and reuse it across messages.
    """
    token_rate = _resolve_param(payload, "token_rate", DEFAULT_TOKEN_RATE, VALID_TOKEN_RATES)
    semantic_vocab_size = _resolve_param(
        payload, "semantic_vocab_size", DEFAULT_SEMANTIC_VOCAB_SIZE, VALID_VOCAB_SIZES
    )
    sample_rate = int(payload.get("sample_rate") or SAMPLE_RATE)
    return token_rate, semantic_vocab_size, sample_rate


def decode_comm_payload(
    payload: dict,
    *,
    token_rate: int | None = None,
    semantic_vocab_size: int | None = None,
    sample_rate: int | None = None,
    output_dir: str | Path = "output",
    gain: float = 1.5,
    device: str = "cpu",
    cache_dir: str | None = None,
    ddim_steps: int = 50,
    cfg_scale: float = 2.0,
    model=None,
) -> DecodeResult:
    """Decode a Comunicação JSON message into ``output/<message_id>.wav``.

    ``token_rate``/``semantic_vocab_size``/``sample_rate`` are resolved from the
    payload when not given explicitly. Pass ``model`` (plus the matching params)
    to reuse a previously loaded decoder (see the watcher).
    """
    if not payload.get("message_id"):
        raise AdapterError("Payload missing 'message_id'.")
    message_id = str(payload["message_id"])

    tokens = tokens_to_tensor(payload.get("tokens"))
    if token_rate is None or semantic_vocab_size is None or sample_rate is None:
        r_token_rate, r_vocab, r_sample_rate = resolve_params(payload)
        token_rate = r_token_rate if token_rate is None else token_rate
        semantic_vocab_size = r_vocab if semantic_vocab_size is None else semantic_vocab_size
        sample_rate = r_sample_rate if sample_rate is None else sample_rate

    return decode_tokens(
        tokens,
        message_id=message_id,
        token_rate=token_rate,
        semantic_vocab_size=semantic_vocab_size,
        sample_rate=sample_rate,
        output_dir=output_dir,
        gain=gain,
        device=device,
        cache_dir=cache_dir,
        ddim_steps=ddim_steps,
        cfg_scale=cfg_scale,
        model=model,
    )


def read_comm_payload(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise AdapterError(f"Payload file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(data, dict):
        raise AdapterError("Payload must be a JSON object.")
    return data


def decode_comm_file(path: str | Path, **kwargs) -> DecodeResult:
    """Read a Comunicação JSON file and decode it."""
    return decode_comm_payload(read_comm_payload(path), **kwargs)
