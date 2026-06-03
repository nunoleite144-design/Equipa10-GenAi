#!/usr/bin/env python3
"""Adaptador do formato JSON da Comunicação para o decode do SemantiCodec.

A Comunicação entrega um JSON por mensagem com os tokens já em claro (lista
``[[semantico, acustico], ...]`` ou achatada). Convertem-se num tensor
``[1, N, 2]`` e chama-se o decode em ``semantic_receiver``. O campo ``text``
(transcrição Whisper) é ignorado: serve apenas para visualização.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from semantic_payload import SAMPLE_RATE
from semantic_receiver import DecodeResult, decode_tokens

# Parâmetros usados quando o JSON não os traz (o caso normal). Alterar aqui se o
# encode passar a usar outros valores.
DEFAULT_TOKEN_RATE = 100
DEFAULT_SEMANTIC_VOCAB_SIZE = 16384

VALID_TOKEN_RATES = (25, 50, 100)
VALID_VOCAB_SIZES = (4096, 8192, 16384, 32768)


class AdapterError(Exception):
    """Erro de payload da Comunicação malformado."""


def tokens_to_tensor(raw_tokens: Any) -> torch.Tensor:
    """Converte a lista de tokens num tensor ``[1, N, 2]``.

    Aceita ``[[s, a], ...]``, ``[[[s, a], ...]]`` ou a lista achatada
    ``[s, a, s, a, ...]`` (reagrupada em pares).
    """
    if raw_tokens is None:
        raise AdapterError("Payload has no 'tokens'.")
    try:
        tokens = torch.tensor(raw_tokens, dtype=torch.long)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"Could not parse 'tokens' into a tensor: {exc}") from exc

    if tokens.numel() == 0:
        raise AdapterError("Payload 'tokens' is empty.")

    if tokens.dim() == 1:
        # Lista achatada [s, a, s, a, ...] -> reagrupar em pares [N, 2].
        if tokens.shape[0] % 2 != 0:
            raise AdapterError(
                f"Flat 'tokens' has odd length {tokens.shape[0]}; expected an even "
                f"number of values (row-major [semantic, acoustic] pairs)."
            )
        tokens = tokens.reshape(-1, 2)

    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)
    elif tokens.dim() != 3:
        raise AdapterError(
            f"Unexpected tokens shape {list(tokens.shape)}; "
            f"expected flat [2N], [N, 2] or [1, N, 2]."
        )

    if tokens.shape[-1] != 2:
        raise AdapterError(
            f"SemantiCodec expects 2 columns (semantic, acoustic); "
            f"got {tokens.shape[-1]} in shape {list(tokens.shape)}."
        )
    return tokens


def _resolve_param(payload: dict, key: str, default: int, valid: tuple[int, ...]) -> int:
    """Devolve o parâmetro: o valor do payload se presente e válido, senão o ``default``.

    A ausência é o caso normal (silenciosa); um valor presente mas inválido gera erro.
    """
    value = payload.get(key)
    if value is None:
        return default
    value = int(value)
    if value not in valid:
        raise AdapterError(f"'{key}' must be one of {valid}; got {value}.")
    return value


def resolve_params(payload: dict) -> tuple[int, int, int]:
    """Devolve ``(token_rate, semantic_vocab_size, sample_rate)`` do payload.

    Usa os valores por defeito quando o payload não os indica.
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
    """Descodifica uma mensagem JSON da Comunicação para ``output/<message_id>.wav``.

    Os parâmetros do codec são lidos do payload quando não forem passados.
    Passar ``model`` para reutilizar um descodificador já carregado.
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
    """Lê um ficheiro JSON da Comunicação e descodifica-o."""
    return decode_comm_payload(read_comm_payload(path), **kwargs)
