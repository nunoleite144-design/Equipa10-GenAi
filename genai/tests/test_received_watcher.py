import json
import os
import time
from pathlib import Path

import received_watcher
from comunicacao_adapter import AdapterError
from semantic_receiver import DecodeResult


def _write_msg(dirpath: Path, name: str, payload: dict) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_result(payload, kwargs):
    mid = str(payload["message_id"])
    return DecodeResult(
        message_id=mid,
        token_rate=kwargs["token_rate"],
        semantic_vocab_size=kwargs["semantic_vocab_size"],
        tokens_shape=[1, 2, 2],
        decode_latency_ms=7,
        gain=kwargs["gain"],
        ddim_steps=kwargs["ddim_steps"],
        audio_file=Path(kwargs["output_dir"]) / f"{mid}.wav",
    )


def test_process_file_success_writes_status_and_moves_to_processed(monkeypatch, tmp_path):
    watch = tmp_path / "received"
    out = tmp_path / "out"
    processed = watch / "processed"
    failed = watch / "failed"
    msg = _write_msg(watch, "m1.json", {"message_id": "m1", "tokens": [1, 2, 3, 4], "sample_rate": 16000})

    monkeypatch.setattr(received_watcher, "load_model", lambda **k: "FAKE_MODEL")
    monkeypatch.setattr(received_watcher, "decode_comm_payload", lambda payload, **kw: _fake_result(payload, kw))

    received_watcher.process_file(
        msg,
        output_dir=out,
        processed_dir=processed,
        failed_dir=failed,
        model_cache={},
        device="cpu",
        cache_dir=None,
        ddim_steps=50,
        cfg_scale=2.0,
        gain=1.5,
    )

    # o ficheiro saiu de received/ e foi para processed/
    assert not msg.exists()
    assert (processed / "m1.json").exists()
    # estado escrito na pasta de saída, com status final completed
    status = json.loads((out / "m1_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["audio_file"].endswith("m1.wav")


def test_process_file_decode_error_moves_to_failed(monkeypatch, tmp_path):
    watch = tmp_path / "received"
    out = tmp_path / "out"
    processed = watch / "processed"
    failed = watch / "failed"
    msg = _write_msg(watch, "m2.json", {"message_id": "m2", "tokens": [1, 2, 3, 4], "sample_rate": 16000})

    monkeypatch.setattr(received_watcher, "load_model", lambda **k: "FAKE_MODEL")

    def boom(payload, **kwargs):
        raise RuntimeError("decode kaput")

    monkeypatch.setattr(received_watcher, "decode_comm_payload", boom)

    received_watcher.process_file(
        msg,
        output_dir=out,
        processed_dir=processed,
        failed_dir=failed,
        model_cache={},
        device="cpu",
        cache_dir=None,
        ddim_steps=50,
        cfg_scale=2.0,
        gain=1.5,
    )

    assert not msg.exists()
    assert (failed / "m2.json").exists()
    status = json.loads((out / "m2_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["stage"] == "decode"


def test_process_file_adapter_error_moves_to_failed(monkeypatch, tmp_path):
    watch = tmp_path / "received"
    out = tmp_path / "out"
    processed = watch / "processed"
    failed = watch / "failed"
    msg = _write_msg(watch, "m3.json", {"message_id": "m3", "tokens": [1, 2, 3, 4], "sample_rate": 16000})

    monkeypatch.setattr(received_watcher, "load_model", lambda **k: "FAKE_MODEL")

    def bad(payload, **kwargs):
        raise AdapterError("bad tokens")

    monkeypatch.setattr(received_watcher, "decode_comm_payload", bad)

    received_watcher.process_file(
        msg,
        output_dir=out,
        processed_dir=processed,
        failed_dir=failed,
        model_cache={},
        device="cpu",
        cache_dir=None,
        ddim_steps=50,
        cfg_scale=2.0,
        gain=1.5,
    )

    assert not msg.exists()
    assert (failed / "m3.json").exists()
    status = json.loads((out / "m3_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["stage"] == "payload_validation"


def test_watch_once_ignores_subdirs_and_status_files(monkeypatch, tmp_path):
    watch = tmp_path / "received"
    watch.mkdir()
    out = tmp_path / "out"

    # uma mensagem pendente legítima
    msg = _write_msg(watch, "20260101T000000_abc.json", {"message_id": "abc", "tokens": [1, 2, 3, 4], "sample_rate": 16000})
    # um ficheiro já em processed/ não deve ser revisto (glob não recursivo)
    _write_msg(watch / "processed", "old.json", {"message_id": "old"})
    # um ficheiro de estado na pasta vigiada deve ser ignorado
    stale_status = watch / "abc_status.json"
    stale_status.write_text("{}", encoding="utf-8")

    # tornar os ficheiros antigos o suficiente para passar a janela de estabilidade
    old = time.time() - 10
    for f in (msg, stale_status):
        os.utime(f, (old, old))

    monkeypatch.setattr(received_watcher, "load_model", lambda **k: "FAKE_MODEL")
    monkeypatch.setattr(received_watcher, "decode_comm_payload", lambda payload, **kw: _fake_result(payload, kw))

    received_watcher.watch(
        watch,
        out,
        poll_seconds=0.0,
        run_once=True,
        device="cpu",
        cache_dir=None,
        ddim_steps=50,
        cfg_scale=2.0,
        gain=1.5,
    )

    # mensagem pendente descodificada e movida
    assert not msg.exists()
    assert (watch / "processed" / msg.name).exists()
    assert (out / "abc_status.json").exists()
    # o ficheiro já arquivado e o de estado na pasta vigiada ficaram intactos
    assert (watch / "processed" / "old.json").exists()
    assert stale_status.exists()
