#!/usr/bin/env python3
"""Reconstrói áudio a partir de um payload JSON do SemantiCodec."""

from __future__ import annotations

import argparse

from semantic_receiver import decode_payload_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receiver SemantiCodec")
    parser.add_argument("payload", help="Payload JSON criado pelo semantic_transmitter_demo.py")
    parser.add_argument("--output-dir", default="output", help="Pasta para o áudio reconstruído")
    parser.add_argument("--gain", type=float, default=1.5, help="Ganho aplicado após normalização")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Dispositivo Torch. Por defeito cpu, pois o decode pode falhar em MPS (Mac).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Pasta de cache dos checkpoints.",
    )
    parser.add_argument("--ddim-steps", type=int, default=50, help="Passos de amostragem do decoder. Menos passos = mais rápido, menor qualidade.")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Escala de guidance do decoder.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = decode_payload_file(
        args.payload,
        output_dir=args.output_dir,
        gain=args.gain,
        device=args.device,
        cache_dir=args.cache_dir,
        ddim_steps=args.ddim_steps,
        cfg_scale=args.cfg_scale,
    )

    print("Áudio reconstruído com sucesso")
    print(f"  message_id: {result.message_id}")
    print(f"  token_rate: {result.token_rate}")
    print(f"  semantic_vocab_size: {result.semantic_vocab_size}")
    print(f"  tokens_shape: {result.tokens_shape}")
    print(f"  decode_latency_ms: {result.decode_latency_ms}")
    print(f"  gain: {result.gain}")
    print(f"  ddim_steps: {result.ddim_steps}")
    print(f"  audio_file: {result.audio_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
