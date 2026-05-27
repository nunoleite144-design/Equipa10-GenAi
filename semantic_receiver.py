"""Reusable SemantiCodec receiver helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from semantic_payload import (
    SAMPLE_RATE,
    read_payload,
    tokens_from_base64,
    validate_payload,
)


@dataclass(frozen=True)
class DecodeResult:
    message_id: str
    token_rate: int
    semantic_vocab_size: int
    tokens_shape: list[int]
    decode_latency_ms: int
    gain: float
    ddim_steps: int
    audio_file: Path


def normalize_audio(waveform: torch.Tensor | np.ndarray, gain: float) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        audio = waveform.squeeze().detach().cpu().numpy()
    else:
        audio = np.asarray(waveform).squeeze()

    max_amplitude = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_amplitude > 0:
        audio = audio / max_amplitude

    audio = audio * gain
    return np.clip(audio, -1.0, 1.0)


def _create_model(
    *,
    token_rate: int,
    semantic_vocab_size: int,
    device: str,
    cache_dir: str | None,
    ddim_steps: int,
    cfg_scale: float,
):
    from semanticodec import SemantiCodec

    return SemantiCodec(
        token_rate=token_rate,
        semantic_vocab_size=semantic_vocab_size,
        device=None if device == "auto" else device,
        cache_path=cache_dir,
        ddim_sample_step=ddim_steps,
        cfg_scale=cfg_scale,
    )


def load_model(
    *,
    token_rate: int,
    semantic_vocab_size: int,
    device: str = "cpu",
    cache_dir: str | None = None,
    ddim_steps: int = 50,
    cfg_scale: float = 2.0,
):
    """Build a SemantiCodec decoder.

    Exposed so long-running consumers (e.g. the received/ watcher) can build the
    model once and reuse it across many messages instead of reloading the
    checkpoints on every decode.
    """
    return _create_model(
        token_rate=token_rate,
        semantic_vocab_size=semantic_vocab_size,
        device=device,
        cache_dir=cache_dir,
        ddim_steps=ddim_steps,
        cfg_scale=cfg_scale,
    )


def decode_tokens(
    tokens: torch.Tensor,
    *,
    message_id: str,
    token_rate: int,
    semantic_vocab_size: int,
    sample_rate: int = SAMPLE_RATE,
    output_dir: str | Path = "output",
    gain: float = 1.5,
    device: str = "cpu",
    cache_dir: str | None = None,
    ddim_steps: int = 50,
    cfg_scale: float = 2.0,
    model=None,
) -> DecodeResult:
    """Reconstruct audio from a ready SemantiCodec token tensor.

    This is the shared decode core. ``tokens`` must already be a tensor with
    shape ``[batch, length, 2]`` (e.g. ``[1, N, 2]``). Pass ``model`` to reuse a
    previously loaded decoder; otherwise one is built on demand.
    """
    if model is None:
        print("Preparing SemantiCodec decoder...")
        model = load_model(
            token_rate=token_rate,
            semantic_vocab_size=semantic_vocab_size,
            device=device,
            cache_dir=cache_dir,
            ddim_steps=ddim_steps,
            cfg_scale=cfg_scale,
        )
    tokens = tokens.to(model.device)

    started = time.perf_counter()
    with torch.no_grad():
        reconstructed_waveform = model.decode(tokens)
    decode_ms = int((time.perf_counter() - started) * 1000)

    audio = normalize_audio(reconstructed_waveform, gain=gain)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{message_id}.wav"
    sf.write(output_path, audio, int(sample_rate))

    return DecodeResult(
        message_id=str(message_id),
        token_rate=token_rate,
        semantic_vocab_size=semantic_vocab_size,
        tokens_shape=list(tokens.shape),
        decode_latency_ms=decode_ms,
        gain=gain,
        ddim_steps=ddim_steps,
        audio_file=output_path,
    )


def decode_payload(
    payload: dict,
    output_dir: str | Path = "output",
    gain: float = 1.5,
    device: str = "cpu",
    cache_dir: str | None = None,
    ddim_steps: int = 50,
    cfg_scale: float = 2.0,
) -> DecodeResult:
    """Decode a canonical SemantiCodec payload (torch_pt_base64 tokens)."""
    validate_payload(payload)

    tokens = tokens_from_base64(payload["tokens"])
    return decode_tokens(
        tokens,
        message_id=str(payload["message_id"]),
        token_rate=int(payload["token_rate"]),
        semantic_vocab_size=int(payload["semantic_vocab_size"]),
        sample_rate=int(payload["sample_rate"]),
        output_dir=output_dir,
        gain=gain,
        device=device,
        cache_dir=cache_dir,
        ddim_steps=ddim_steps,
        cfg_scale=cfg_scale,
    )


def decode_payload_file(
    payload_path: str | Path,
    output_dir: str | Path = "output",
    gain: float = 1.5,
    device: str = "cpu",
    cache_dir: str | None = None,
    ddim_steps: int = 50,
    cfg_scale: float = 2.0,
) -> DecodeResult:
    payload = read_payload(Path(payload_path))
    return decode_payload(
        payload,
        output_dir=output_dir,
        gain=gain,
        device=device,
        cache_dir=cache_dir,
        ddim_steps=ddim_steps,
        cfg_scale=cfg_scale,
    )
