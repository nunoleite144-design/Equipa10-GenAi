#!/usr/bin/env python3
"""Run the full local SemantiCodec demo: audio -> JSON payload -> reconstructed audio."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SemantiCodec transmitter and receiver demos")
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("--message-id", default="msg-001")
    parser.add_argument("--gain", type=float, default=1.5)
    parser.add_argument("--receiver-device", default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--transmitter-device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--ddim-steps", type=int, default=50)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    payload = f"payloads/{args.message_id}.json"

    run(
        [
            sys.executable,
            "semantic_transmitter_demo.py",
            args.audio,
            "--message-id",
            args.message_id,
            "--device",
            args.transmitter_device,
        ]
    )
    run(
        [
            sys.executable,
            "semantic_receiver_demo.py",
            payload,
            "--device",
            args.receiver_device,
            "--gain",
            str(args.gain),
            "--ddim-steps",
            str(args.ddim_steps),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
