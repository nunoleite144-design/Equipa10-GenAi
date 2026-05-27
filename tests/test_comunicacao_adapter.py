from pathlib import Path

import pytest
import torch

import comunicacao_adapter
from comunicacao_adapter import (
    DEFAULT_SEMANTIC_VOCAB_SIZE,
    DEFAULT_TOKEN_RATE,
    AdapterError,
    decode_comm_payload,
    resolve_params,
    tokens_to_tensor,
)
from semantic_receiver import DecodeResult


def test_tokens_to_tensor_2d_list_gets_batch_dim():
    tokens = tokens_to_tensor([[1, 2], [3, 4], [5, 6]])
    assert tokens.shape == (1, 3, 2)
    assert tokens.dtype == torch.long


def test_tokens_to_tensor_passthrough_when_already_batched():
    tokens = tokens_to_tensor([[[1, 2], [3, 4]]])
    assert tokens.shape == (1, 2, 2)


def test_tokens_to_tensor_rejects_wrong_columns():
    with pytest.raises(AdapterError):
        tokens_to_tensor([[1, 2, 3], [4, 5, 6]])


def test_tokens_to_tensor_rejects_empty_and_none():
    with pytest.raises(AdapterError):
        tokens_to_tensor([])
    with pytest.raises(AdapterError):
        tokens_to_tensor(None)


def test_resolve_params_reads_from_payload():
    payload = {"token_rate": 50, "semantic_vocab_size": 8192, "sample_rate": 16000}
    assert resolve_params(payload) == (50, 8192, 16000)


def test_resolve_params_falls_back_to_defaults_when_missing():
    token_rate, vocab, sample_rate = resolve_params({})
    assert token_rate == DEFAULT_TOKEN_RATE
    assert vocab == DEFAULT_SEMANTIC_VOCAB_SIZE
    assert sample_rate == 16000


def test_resolve_params_rejects_invalid_values():
    with pytest.raises(AdapterError):
        resolve_params({"token_rate": 7, "semantic_vocab_size": 16384})


def test_decode_comm_payload_requires_message_id():
    with pytest.raises(AdapterError):
        decode_comm_payload({"tokens": [[1, 2]]})


def test_decode_comm_payload_bridges_to_core_and_ignores_text(monkeypatch):
    captured = {}

    def fake_decode_tokens(tokens, **kwargs):
        captured["tokens"] = tokens
        captured["kwargs"] = kwargs
        return DecodeResult(
            message_id=kwargs["message_id"],
            token_rate=kwargs["token_rate"],
            semantic_vocab_size=kwargs["semantic_vocab_size"],
            tokens_shape=list(tokens.shape),
            decode_latency_ms=1,
            gain=kwargs["gain"],
            ddim_steps=kwargs["ddim_steps"],
            audio_file=Path(kwargs["output_dir"]) / f"{kwargs['message_id']}.wav",
        )

    monkeypatch.setattr(comunicacao_adapter, "decode_tokens", fake_decode_tokens)

    payload = {
        "message_id": "abc",
        "tokens": [[10, 20], [30, 40]],
        "text": "este texto do Whisper nao deve entrar no decode",
        "token_rate": 100,
        "semantic_vocab_size": 16384,
        "sample_rate": 16000,
    }

    result = decode_comm_payload(payload, output_dir="out")

    assert result.message_id == "abc"
    assert captured["tokens"].shape == (1, 2, 2)
    assert captured["kwargs"]["token_rate"] == 100
    assert captured["kwargs"]["semantic_vocab_size"] == 16384
    # text must never be forwarded to the decode core
    assert "text" not in captured["kwargs"]
