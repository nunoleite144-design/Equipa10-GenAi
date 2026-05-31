#!/usr/bin/env python3
"""Watch a received/ folder for SemantiCodec JSON messages and reconstruct audio.

This is the GenAI live entry point for the file-based handoff: the
Comunicação module subscribes to MQTT, decrypts each message and writes a JSON
file into a ``received/`` directory. This watcher picks each new file up and
writes ``output/<message_id>.wav`` plus ``output/<message_id>_status.json``.

The decoder is loaded once per ``(token_rate, semantic_vocab_size)`` and reused
across messages, so a live demo does not reload the multi-GB checkpoints on
every message.

Run:
    python received_watcher.py
    python received_watcher.py --watch-dir comunicação/received --once
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from comunicacao_adapter import (
    AdapterError,
    decode_comm_payload,
    read_comm_payload,
    resolve_params,
)
from semantic_receiver import load_model

DEFAULT_WATCH_DIR = Path("comunicação") / "received"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_POLL_SECONDS = 1.0
# Only process files untouched for at least this long, so we never read a JSON
# while Comunicação is still writing it.
STABILITY_SECONDS = 0.5

CODEC_NAME = "semanticodec"


def write_status(output_dir: Path, message_id: str, status: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{message_id}_status.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def process_file(
    path: Path,
    *,
    output_dir: Path,
    model_cache: dict,
    device: str,
    cache_dir: str | None,
    ddim_steps: int,
    cfg_scale: float,
    gain: float,
) -> None:
    """Decode a single received JSON file; record completed/failed status."""
    message_id = path.stem
    try:
        payload = read_comm_payload(path)
        message_id = str(payload.get("message_id") or path.stem)
        write_status(
            output_dir,
            message_id,
            {"message_id": message_id, "status": "decoding", "source": path.name},
        )

        token_rate, vocab, sample_rate = resolve_params(payload)
        key = (token_rate, vocab)
        model = model_cache.get(key)
        if model is None:
            print(f"[watcher] loading SemantiCodec (token_rate={token_rate}, vocab={vocab}) ...")
            model = load_model(
                token_rate=token_rate,
                semantic_vocab_size=vocab,
                device=device,
                cache_dir=cache_dir,
                ddim_steps=ddim_steps,
                cfg_scale=cfg_scale,
            )
            model_cache[key] = model

        result = decode_comm_payload(
            payload,
            token_rate=token_rate,
            semantic_vocab_size=vocab,
            sample_rate=sample_rate,
            output_dir=output_dir,
            gain=gain,
            device=device,
            cache_dir=cache_dir,
            ddim_steps=ddim_steps,
            cfg_scale=cfg_scale,
            model=model,
        )

        write_status(
            output_dir,
            message_id,
            {
                "message_id": result.message_id,
                "status": "completed",
                "audio_file": str(result.audio_file),
                "codec": CODEC_NAME,
                "sample_rate": sample_rate,
                "token_rate": result.token_rate,
                "semantic_vocab_size": result.semantic_vocab_size,
                "tokens_shape": result.tokens_shape,
                "decode_latency_ms": result.decode_latency_ms,
                "ddim_steps": result.ddim_steps,
                "gain": result.gain,
            },
        )
        print(f"[watcher] OK {path.name} -> {result.audio_file} ({result.decode_latency_ms} ms)")
    except AdapterError as exc:
        write_status(
            output_dir,
            message_id,
            {
                "message_id": message_id,
                "status": "failed",
                "stage": "payload_validation",
                "error_code": "invalid_payload",
                "error_message": str(exc),
            },
        )
        print(f"[watcher] FAILED {path.name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - keep the watcher alive on any decode error
        write_status(
            output_dir,
            message_id,
            {
                "message_id": message_id,
                "status": "failed",
                "stage": "decode",
                "error_code": "decode_error",
                "error_message": str(exc),
            },
        )
        print(f"[watcher] ERROR {path.name}: {exc}")


def watch(
    watch_dir: Path,
    output_dir: Path,
    *,
    poll_seconds: float,
    run_once: bool,
    device: str,
    cache_dir: str | None,
    ddim_steps: int,
    cfg_scale: float,
    gain: float,
) -> None:
    watch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed: set[str] = set()
    model_cache: dict = {}

    print(f"[watcher] watching {watch_dir.resolve()} -> {output_dir.resolve()}")
    if run_once:
        print("[watcher] single pass (--once)")
    else:
        print("[watcher] live mode; Ctrl+C to stop")

    while True:
        now = time.time()
        for path in sorted(watch_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            if path.name in processed:
                continue
            # Skip files still being written (mtime too recent).
            if now - path.stat().st_mtime < STABILITY_SECONDS:
                continue
            process_file(
                path,
                output_dir=output_dir,
                model_cache=model_cache,
                device=device,
                cache_dir=cache_dir,
                ddim_steps=ddim_steps,
                cfg_scale=cfg_scale,
                gain=gain,
            )
            processed.add(path.name)

        if run_once:
            break
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GenAI received/ watcher (SemantiCodec)")
    parser.add_argument("--watch-dir", default=str(DEFAULT_WATCH_DIR), help="Folder Comunicação writes JSON messages into")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to write .wav and status files")
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_SECONDS, help="Seconds between folder scans")
    parser.add_argument("--once", action="store_true", help="Process current files once and exit (no live loop)")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda", "mps"), help="Torch device")
    parser.add_argument("--cache-dir", default=None, help="SemantiCodec checkpoint cache directory")
    parser.add_argument("--ddim-steps", type=int, default=50, help="Decoder sampling steps. Lower is faster, may reduce quality")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Decoder guidance scale")
    parser.add_argument("--gain", type=float, default=1.5, help="Post-normalization gain")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        watch(
            Path(args.watch_dir),
            Path(args.output_dir),
            poll_seconds=args.poll,
            run_once=args.once,
            device=args.device,
            cache_dir=args.cache_dir,
            ddim_steps=args.ddim_steps,
            cfg_scale=args.cfg_scale,
            gain=args.gain,
        )
    except KeyboardInterrupt:
        print("\n[watcher] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
