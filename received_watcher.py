#!/usr/bin/env python3
"""Vigia uma pasta ``received/`` e reconstrói o áudio de cada mensagem nova.

Para cada JSON novo escreve ``output/<message_id>.wav`` e
``output/<message_id>_status.json``. O descodificador é carregado uma vez por
``(token_rate, semantic_vocab_size)`` e reutilizado entre mensagens.

Uso:
    python received_watcher.py
    python received_watcher.py --watch-dir ../comunicacao/received --once
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from comunicacao_adapter import (
    AdapterError,
    decode_comm_payload,
    read_comm_payload,
    resolve_params,
)
from semantic_receiver import load_model

# Estrutura de deploy: projeto/{<genai>, comunicacao/received, webapp}, com este
# repo e comunicacao/ como pastas irmãs. Caminho relativo ao repo; o launcher
# passa um caminho absoluto e --watch-dir sobrepõe-no.
DEFAULT_WATCH_DIR = Path("..") / "comunicacao" / "received"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_POLL_SECONDS = 1.0
# Só processa ficheiros parados há pelo menos este tempo, para não ler um JSON
# enquanto a Comunicação ainda o está a escrever.
STABILITY_SECONDS = 0.5
# Subpastas para onde vão os ficheiros já tratados, para received/ ficar só com os
# pendentes e um reinício não reprocessar nada.
PROCESSED_SUBDIR = "processed"
FAILED_SUBDIR = "failed"

CODEC_NAME = "semanticodec"


def write_status(output_dir: Path, message_id: str, status: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{message_id}_status.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def archive_input(path: Path, dest_dir: Path) -> None:
    """Move um JSON já tratado para fora da pasta vigiada (ignora erros ao mover)."""
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / path.name
        if target.exists():
            target.unlink()
        shutil.move(str(path), str(target))
    except Exception as exc:  # noqa: BLE001
        print(f"[watcher] AVISO: não consegui mover {path.name} -> {dest_dir.name}/: {exc}")


def process_file(
    path: Path,
    *,
    output_dir: Path,
    processed_dir: Path,
    failed_dir: Path,
    model_cache: dict,
    device: str,
    cache_dir: str | None,
    ddim_steps: int,
    cfg_scale: float,
    gain: float,
) -> None:
    """Descodifica um JSON; grava o estado e move o ficheiro para processed/ (sucesso) ou failed/ (erro)."""
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
            print(f"[watcher] a carregar SemantiCodec (token_rate={token_rate}, vocab={vocab}) ...")
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
        archive_input(path, processed_dir)
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
        print(f"[watcher] FALHA {path.name}: {exc}")
        archive_input(path, failed_dir)
    except Exception as exc:  # noqa: BLE001
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
        print(f"[watcher] ERRO {path.name}: {exc}")
        archive_input(path, failed_dir)


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
    processed_dir = watch_dir / PROCESSED_SUBDIR
    failed_dir = watch_dir / FAILED_SUBDIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    processed: set[str] = set()
    model_cache: dict = {}

    print(f"[watcher] a vigiar {watch_dir.resolve()} -> {output_dir.resolve()}")
    print(f"[watcher] processados -> {processed_dir.name}/ , erros -> {failed_dir.name}/")
    if run_once:
        print("[watcher] passagem única (--once)")
    else:
        print("[watcher] modo contínuo; Ctrl+C para parar")

    while True:
        now = time.time()
        # glob não é recursivo, por isso processed/ e failed/ não são revistos.
        for path in sorted(watch_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            if path.name in processed:
                continue
            # Não tratar ficheiros de estado como mensagens de entrada.
            if path.name.endswith("_status.json"):
                continue
            # Saltar ficheiros ainda a ser escritos (mtime demasiado recente).
            if now - path.stat().st_mtime < STABILITY_SECONDS:
                continue
            process_file(
                path,
                output_dir=output_dir,
                processed_dir=processed_dir,
                failed_dir=failed_dir,
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
    parser = argparse.ArgumentParser(description="Watcher da pasta received/ (SemantiCodec)")
    parser.add_argument("--watch-dir", default=str(DEFAULT_WATCH_DIR), help="Pasta onde a Comunicação grava os JSON")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Pasta para os .wav e ficheiros de estado")
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_SECONDS, help="Segundos entre varrimentos da pasta")
    parser.add_argument("--once", action="store_true", help="Processa os ficheiros atuais uma vez e termina")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda", "mps"), help="Dispositivo Torch")
    parser.add_argument("--cache-dir", default=None, help="Pasta de cache dos checkpoints")
    parser.add_argument("--ddim-steps", type=int, default=50, help="Passos de amostragem do decoder. Menos = mais rápido, menor qualidade")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Escala de guidance do decoder")
    parser.add_argument("--gain", type=float, default=1.5, help="Ganho aplicado após normalização")
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
        print("\n[watcher] terminado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
