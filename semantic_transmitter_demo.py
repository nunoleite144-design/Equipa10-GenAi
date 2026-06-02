#!/usr/bin/env python3
"""Codifica áudio num payload JSON do SemantiCodec."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from semantic_payload import build_payload, write_payload
from semanticodec import SemantiCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transmitter SemantiCodec")
    parser.add_argument("audio", help="Ficheiro de áudio de entrada, de preferência um .wav curto")
    parser.add_argument("--message-id", default="msg-001", help="Identificador da mensagem")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Dispositivo Torch. Usar cpu se mps/cuda der problemas.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Pasta de cache dos checkpoints.",
    )
    parser.add_argument("--token-rate", type=int, default=100, choices=(25, 50, 100))
    parser.add_argument(
        "--semantic-vocab-size",
        type=int,
        default=16384,
        choices=(4096, 8192, 16384, 32768),
    )
    parser.add_argument("--output-dir", default="payloads", help="Pasta para os payloads JSON")
    parser.add_argument("--pretty-json", action="store_true", help="Escreve JSON legível em vez de compacto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Erro: ficheiro de áudio não encontrado: {audio_path}")

    print("A preparar o encoder SemantiCodec...")
    model = SemantiCodec(
        token_rate=args.token_rate,
        semantic_vocab_size=args.semantic_vocab_size,
        device=None if args.device == "auto" else args.device,
        cache_path=args.cache_dir,
    )

    started = time.perf_counter()
    with torch.no_grad():
        tokens = model.encode(str(audio_path))
    encode_ms = int((time.perf_counter() - started) * 1000)

    payload = build_payload(
        message_id=args.message_id,
        tokens=tokens,
        token_rate=args.token_rate,
        semantic_vocab_size=args.semantic_vocab_size,
        source_audio=str(audio_path),
    )

    output_path = Path(args.output_dir) / f"{args.message_id}.json"
    write_payload(output_path, payload, pretty=args.pretty_json)
    size_kb = output_path.stat().st_size / 1024

    print("Payload criado com sucesso")
    print(f"  message_id: {args.message_id}")
    print(f"  tokens_shape: {list(tokens.shape)}")
    print(f"  encode_latency_ms: {encode_ms}")
    print(f"  payload_file: {output_path}")
    print(f"  payload_size_kb: {size_kb:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
