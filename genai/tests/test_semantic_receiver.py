from pathlib import Path

import numpy as np
import pytest
import torch

import semantic_receiver
from semantic_payload import build_payload
from semantic_receiver import DecodeResult, decode_payload, decode_payload_file, normalize_audio


@pytest.fixture
def sample_tokens():
    return torch.tensor([[[1], [2], [3], [4]]], dtype=torch.long)


@pytest.fixture
def valid_payload(sample_tokens):
    return build_payload(
        message_id="msg-001",
        tokens=sample_tokens,
        token_rate=100,
        semantic_vocab_size=16384,
    )


class FakeSemantiCodec:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.decoded_tokens = None

    def decode(self, tokens):
        self.decoded_tokens = tokens
        return torch.tensor([[[-0.5, 0.0, 0.5]]], dtype=torch.float32)


def test_normalize_audio_normalizes_gain_and_clips_tensor():
    waveform = torch.tensor([[[-2.0, 0.0, 1.0, 2.0]]])

    audio = normalize_audio(waveform, gain=1.5)

    np.testing.assert_allclose(audio, np.array([-1.0, 0.0, 0.75, 1.0]))


def test_normalize_audio_accepts_numpy_array():
    waveform = np.array([[-0.25, 0.0, 0.25]])

    audio = normalize_audio(waveform, gain=1.0)

    np.testing.assert_allclose(audio, np.array([-1.0, 0.0, 1.0]))


def test_decode_payload_decodes_writes_audio_and_returns_result(monkeypatch, tmp_path, valid_payload):
    created_models = []
    writes = []

    def fake_create_model(**kwargs):
        created_models.append(kwargs)
        return FakeSemantiCodec()

    def fake_write(path, audio, sample_rate):
        writes.append((Path(path), np.asarray(audio), sample_rate))

    monkeypatch.setattr(semantic_receiver, "_create_model", fake_create_model)
    monkeypatch.setattr(semantic_receiver.sf, "write", fake_write)

    result = decode_payload(
        valid_payload,
        output_dir=tmp_path,
        gain=1.25,
        device="auto",
        cache_dir="cache",
        ddim_steps=25,
        cfg_scale=1.5,
    )

    assert isinstance(result, DecodeResult)
    assert result.message_id == "msg-001"
    assert result.token_rate == 100
    assert result.semantic_vocab_size == 16384
    assert result.tokens_shape == [1, 4, 1]
    assert result.gain == 1.25
    assert result.ddim_steps == 25
    assert result.audio_file == tmp_path / "msg-001.wav"
    assert result.decode_latency_ms >= 0

    assert created_models == [
        {
            "token_rate": 100,
            "semantic_vocab_size": 16384,
            "device": "auto",
            "cache_dir": "cache",
            "ddim_steps": 25,
            "cfg_scale": 1.5,
        }
    ]
    assert len(writes) == 1
    output_path, audio, sample_rate = writes[0]
    assert output_path == tmp_path / "msg-001.wav"
    assert sample_rate == 16000
    np.testing.assert_allclose(audio, np.array([-1.0, 0.0, 1.0]))


def test_decode_payload_file_reads_payload_and_delegates(monkeypatch, tmp_path, valid_payload):
    payload_path = tmp_path / "payload.json"
    delegated = []

    def fake_read_payload(path):
        assert path == payload_path
        return valid_payload

    def fake_decode_payload(payload, **kwargs):
        delegated.append((payload, kwargs))
        return DecodeResult(
            message_id="msg-001",
            token_rate=100,
            semantic_vocab_size=16384,
            tokens_shape=[1, 4, 1],
            decode_latency_ms=12,
            gain=kwargs["gain"],
            ddim_steps=kwargs["ddim_steps"],
            audio_file=Path(kwargs["output_dir"]) / "msg-001.wav",
        )

    monkeypatch.setattr(semantic_receiver, "read_payload", fake_read_payload)
    monkeypatch.setattr(semantic_receiver, "decode_payload", fake_decode_payload)

    result = decode_payload_file(
        payload_path,
        output_dir=tmp_path / "output",
        gain=1.8,
        device="cpu",
        cache_dir=None,
        ddim_steps=30,
        cfg_scale=2.5,
    )

    assert result.audio_file == tmp_path / "output" / "msg-001.wav"
    assert delegated == [
        (
            valid_payload,
            {
                "output_dir": tmp_path / "output",
                "gain": 1.8,
                "device": "cpu",
                "cache_dir": None,
                "ddim_steps": 30,
                "cfg_scale": 2.5,
            },
        )
    ]
