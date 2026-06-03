# =============================================================================
# publisher.py — Simulador da Equipa 9 (para testes)
# Equipa 10 — ESIST 2025/2026 — Módulo de Comunicação
#
# Uso:
#   python publisher.py --ip <IP_DO_HOST>
#   python publisher.py --ip 192.168.1.42
#   python publisher.py --ip 192.168.1.42 --file tokensdotransmitter1.pt
#   python publisher.py --ip 192.168.1.42 --file tokensdotransmitter1.pt --msg "Texto"
#   python publisher.py --ip 192.168.1.42 --loop
#   python publisher.py --ip 192.168.1.42 --no-crypto   (sem encriptação)
# =============================================================================

import argparse
import json
import time
import uuid
import random
from datetime import datetime, timezone

import torch
import paho.mqtt.client as mqtt

import config
from crypto import encrypt_payload, key_from_hex

# -----------------------------------------------------------------------------
# Mensagens de teste (tokens aleatórios — só para smoke test)
# -----------------------------------------------------------------------------
SAMPLE_MESSAGES = [
    ("Olá, como estás?",          [101, 45, 233, 17, 88, 512, 300, 76]),
    ("O tempo hoje está bom.",     [204, 88, 155, 312, 47, 99, 201, 133]),
    ("Reunião às 14h na sala B.",  [88, 200, 44, 311, 92, 17, 405, 250]),
    ("Sistema operacional.",       [55, 144, 300, 82, 199, 400, 37, 261]),
]


def load_tokens_from_pt(filepath: str) -> list[int]:
    """
    Carrega tokens reais SemanticCodec de um ficheiro .pt.
    Suporta:
      - tensor direto:        torch.save(tensor, "file.pt")
      - dict com 'tokens':    torch.save({"tokens": tensor, ...}, "file.pt")
    """
    data = torch.load(filepath, weights_only=False)

    if isinstance(data, torch.Tensor):
        tokens = data.flatten().tolist()
    elif isinstance(data, dict):
        if "tokens" in data and data["tokens"] is not None:
            tokens = data["tokens"].flatten().tolist() if isinstance(data["tokens"], torch.Tensor) else data["tokens"]
        else:
            raise ValueError(f"Ficheiro .pt não contém campo 'tokens': {list(data.keys())}")
    else:
        raise ValueError(f"Formato desconhecido no ficheiro .pt: {type(data)}")

    # Garantir que são inteiros
    tokens = [int(t) for t in tokens]
    print(f"  [PT] Carregados {len(tokens)} tokens reais  (min={min(tokens)} max={max(tokens)})")
    return tokens


def build_payload(text: str, tokens: list, use_crypto: bool) -> dict:
    payload = {
        "message_id":  str(uuid.uuid4()),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "sample_rate": 16000,
        "language":    "pt",
    }

    if use_crypto:
        key = key_from_hex(config.ENCRYPTION_KEY)
        payload["ciphertext"] = encrypt_payload(tokens, text, key)
    else:
        print("  [AVISO] A enviar tokens SEM encriptação (modo desenvolvimento)")
        payload["tokens"] = tokens
        if text:
            payload["text"] = text

    return payload


def send_one(client, text: str, tokens: list, use_crypto: bool):
    payload = build_payload(text, tokens, use_crypto)
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    client.publish(config.TOPIC_SUBSCRIBE, raw, qos=config.QOS)
    print(f"  → Enviado id={payload['message_id']}  tokens={len(tokens)}  texto='{text}'")


def main():
    parser = argparse.ArgumentParser(description="Simulador da Equipa 9")
    parser.add_argument("--ip",        default="localhost",
                        help="IP do host onde corre o broker (default: localhost)")
    parser.add_argument("--port",      type=int, default=config.BROKER_PORT)
    parser.add_argument("--file",      default=None,
                        help="Ficheiro .pt com tokens reais SemanticCodec (ex: tokensdotransmitter1.pt)")
    parser.add_argument("--msg",       default=None,
                        help="Texto/transcrição a incluir na mensagem")
    parser.add_argument("--loop",      action="store_true",
                        help="Enviar em loop contínuo")
    parser.add_argument("--interval",  type=float, default=3.0,
                        help="Segundos entre mensagens no loop (default: 3)")
    parser.add_argument("--no-crypto", action="store_true",
                        help="Enviar tokens em plaintext (sem encriptação)")
    args = parser.parse_args()

    use_crypto = not args.no_crypto

    print("=" * 55)
    print("  ESIST — Simulador Equipa 9")
    print(f"  Broker : {args.ip}:{args.port}")
    print(f"  Tópico : {config.TOPIC_SUBSCRIBE}")
    print(f"  Crypto : {'AES-256-GCM' if use_crypto else 'DESLIGADA'}")
    if args.file:
        print(f"  Fonte  : {args.file} (tokens reais)")
    else:
        print(f"  Fonte  : tokens de teste (aleatórios)")
    print("=" * 55)

    # Carregar tokens do ficheiro .pt se indicado
    pt_tokens = None
    if args.file:
        pt_tokens = load_tokens_from_pt(args.file)

    client = mqtt.Client(client_id="equipa9_sim")
    client.connect(args.ip, args.port, keepalive=60)
    client.loop_start()
    time.sleep(0.5)

    text = args.msg or ""

    try:
        if args.loop:
            i = 0
            while True:
                if pt_tokens:
                    tokens = pt_tokens
                else:
                    text, tokens = SAMPLE_MESSAGES[i % len(SAMPLE_MESSAGES)]
                send_one(client, text, tokens, use_crypto)
                i += 1
                time.sleep(args.interval)
        else:
            if pt_tokens:
                # Enviar o ficheiro .pt real
                send_one(client, text, pt_tokens, use_crypto)
            elif args.msg:
                tokens = [random.randint(0, 511) for _ in range(12)]
                send_one(client, args.msg, tokens, use_crypto)
            else:
                # Enviar as 4 mensagens de teste
                for text, tokens in SAMPLE_MESSAGES:
                    send_one(client, text, tokens, use_crypto)
                    time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrompido.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Desligado.")


if __name__ == "__main__":
    main()
