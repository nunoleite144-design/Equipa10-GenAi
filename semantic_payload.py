#!/usr/bin/env python3
"""Shared payload helpers for SemantiCodec MQTT-style JSON packages."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import torch


PAYLOAD_TYPE = "semanticodec_tokens"
CODEC_NAME = "semanticodec"
TOKENS_FORMAT = "torch_pt_base64"
SAMPLE_RATE = 16000


class PayloadError(Exception):
    """User-facing payload validation error."""


def tokens_to_base64(tokens: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tokens.detach().cpu(), buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def tokens_from_base64(encoded_tokens: str) -> torch.Tensor:
    try:
        raw = base64.b64decode(encoded_tokens.encode("ascii"))
    except Exception as exc:
        raise PayloadError("Invalid base64 token payload.") from exc

    try:
        return torch.load(io.BytesIO(raw), map_location="cpu")
    except Exception as exc:
        raise PayloadError("Could not load torch tokens from payload.") from exc


def build_payload(
    *,
    message_id: str,
    tokens: torch.Tensor,
    token_rate: int,
    semantic_vocab_size: int,
    source_audio: str | None = None,
) -> dict[str, Any]:
    return {
        "message_id": message_id,
        "type": PAYLOAD_TYPE,
        "codec": CODEC_NAME,
        "token_rate": token_rate,
        "semantic_vocab_size": semantic_vocab_size,
        "sample_rate": SAMPLE_RATE,
        "tokens_format": TOKENS_FORMAT,
        "tokens_shape": list(tokens.shape),
        "source_audio": source_audio,
        "tokens": tokens_to_base64(tokens),
    }


def read_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PayloadError(f"Payload file not found: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PayloadError(f"Invalid JSON payload: {exc}") from exc


def write_payload(path: Path, payload: dict[str, Any], pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        content = json.dumps(payload, indent=2)
    else:
        content = json.dumps(payload, separators=(",", ":"))
    path.write_text(content, encoding="utf-8")


def validate_payload(payload: dict[str, Any]) -> None:
    required = (
        "message_id",
        "type",
        "codec",
        "token_rate",
        "semantic_vocab_size",
        "sample_rate",
        "tokens_format",
        "tokens",
    )
    missing = [field for field in required if field not in payload or payload[field] in ("", None)]
    if missing:
        raise PayloadError(f"Payload missing required field(s): {', '.join(missing)}")

    if payload["type"] != PAYLOAD_TYPE:
        raise PayloadError(f"Invalid payload type: {payload['type']}")
    if payload["codec"] != CODEC_NAME:
        raise PayloadError(f"Invalid codec: {payload['codec']}")
    if payload["tokens_format"] != TOKENS_FORMAT:
        raise PayloadError(f"Invalid tokens_format: {payload['tokens_format']}")
    if payload["sample_rate"] != SAMPLE_RATE:
        raise PayloadError(f"Invalid sample_rate: {payload['sample_rate']}")
    if payload["token_rate"] not in (25, 50, 100):
        raise PayloadError("token_rate must be 25, 50, or 100.")
    if payload["semantic_vocab_size"] not in (4096, 8192, 16384, 32768):
        raise PayloadError("semantic_vocab_size must be 4096, 8192, 16384, or 32768.")
