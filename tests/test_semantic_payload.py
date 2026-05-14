import pytest
import torch

from semantic_payload import (
    CODEC_NAME,
    PAYLOAD_TYPE,
    SAMPLE_RATE,
    TOKENS_FORMAT,
    PayloadError,
    build_payload,
    tokens_from_base64,
    tokens_to_base64,
    validate_payload,
)


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
        source_audio="input.wav",
    )


def test_build_payload_creates_expected_semanticodec_payload(sample_tokens):
    payload = build_payload(
        message_id="msg-001",
        tokens=sample_tokens,
        token_rate=100,
        semantic_vocab_size=16384,
        source_audio="input.wav",
    )

    assert payload["message_id"] == "msg-001"
    assert payload["type"] == PAYLOAD_TYPE
    assert payload["codec"] == CODEC_NAME
    assert payload["token_rate"] == 100
    assert payload["semantic_vocab_size"] == 16384
    assert payload["sample_rate"] == SAMPLE_RATE
    assert payload["tokens_format"] == TOKENS_FORMAT
    assert payload["tokens_shape"] == list(sample_tokens.shape)
    assert payload["source_audio"] == "input.wav"
    assert isinstance(payload["tokens"], str)
    assert payload["tokens"]


def test_tokens_base64_round_trip_preserves_tensor(sample_tokens):
    encoded = tokens_to_base64(sample_tokens)
    decoded = tokens_from_base64(encoded)

    assert torch.equal(decoded, sample_tokens)
    assert decoded.dtype == sample_tokens.dtype
    assert list(decoded.shape) == list(sample_tokens.shape)


def test_validate_payload_accepts_valid_payload(valid_payload):
    validate_payload(valid_payload)


@pytest.mark.parametrize(
    "field",
    [
        "message_id",
        "type",
        "codec",
        "token_rate",
        "semantic_vocab_size",
        "sample_rate",
        "tokens_format",
        "tokens",
    ],
)
def test_validate_payload_rejects_missing_required_fields(valid_payload, field):
    payload = valid_payload.copy()
    payload.pop(field)

    with pytest.raises(PayloadError, match="Payload missing required field"):
        validate_payload(payload)


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_message"),
    [
        ("type", "audio_tokens", "Invalid payload type"),
        ("codec", "other_codec", "Invalid codec"),
        ("sample_rate", 44100, "Invalid sample_rate"),
        ("token_rate", 75, "token_rate must be 25, 50, or 100"),
        (
            "semantic_vocab_size",
            12345,
            "semantic_vocab_size must be 4096, 8192, 16384, or 32768",
        ),
        ("tokens_format", "json_array", "Invalid tokens_format"),
    ],
)
def test_validate_payload_rejects_invalid_contract_values(
    valid_payload, field, invalid_value, expected_message
):
    payload = valid_payload.copy()
    payload[field] = invalid_value

    with pytest.raises(PayloadError, match=expected_message):
        validate_payload(payload)
