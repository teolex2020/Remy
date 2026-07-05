"""
Aura Wallet Checker — перевірка балансу TRX та USDT (TRC-20).

Використання:
    python scripts/check_wallet.py

Потребує:
    pip install tronpy python-dotenv

Конфігурація в .env:
    TRON_PRIVATE_KEY=ваш_приватний_ключ
    WALLET_ADDRESS=TNjyL4vZwBQg1tzudWWM8aFavPCYZTRAJY
    TRONGRID_API_KEY=ваш_ключ (опціонально, для обходу rate limits)
"""

import os
import sys
import time

from dotenv import load_dotenv
from tronpy import Tron
from tronpy.providers import HTTPProvider


load_dotenv()

WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
TRONGRID_API_KEY = os.getenv("TRONGRID_API_KEY")
USDT_CONTRACT = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"

# Endpoints to try (primary + fallbacks)
ENDPOINTS = [
    "https://api.trongrid.io",
    "https://api.tronstack.io",
    "https://api.shasta.trongrid.io",  # testnet fallback
]


def _make_client():
    """Create Tron client with API key if available."""
    headers = {}
    if TRONGRID_API_KEY:
        headers["TRON-PRO-API-KEY"] = TRONGRID_API_KEY

    for endpoint in ENDPOINTS:
        try:
            provider = HTTPProvider(endpoint, timeout=10)
            if headers:
                provider.sess.headers.update(headers)
            client = Tron(provider)
            # Quick check that it works
            client.get_latest_block_number()
            return client
        except Exception:
            continue

    # Last resort: default
    return Tron(HTTPProvider(timeout=10))


def check_balance():
    if not WALLET_ADDRESS:
        print("WALLET_ADDRESS not set in .env")
        sys.exit(1)

    client = _make_client()

    print("--- Гаманець Aura ---")
    print(f"Адреса: {WALLET_ADDRESS}")

    try:
        balance_trx = client.get_account_balance(WALLET_ADDRESS)
        print(f"Баланс TRX:  {balance_trx}")
    except Exception as e:
        if "account not found" in str(e).lower():
            print("Баланс TRX:  0 (гаманець ще не активований)")
            print("\nГаманець існує, але ще не отримував коштів.")
            print("Надішліть будь-яку суму TRX на цю адресу для активації.")
            return
        raise

    time.sleep(1)  # avoid rate limit between calls

    try:
        contract = client.get_contract(USDT_CONTRACT)
        precision = contract.functions.decimals()
        balance_usdt = contract.functions.balanceOf(WALLET_ADDRESS) / (10 ** precision)
        print(f"Баланс USDT: {balance_usdt} $")
    except Exception as e:
        print(f"Баланс USDT: не вдалося перевірити ({e})")

    if balance_trx < 30:
        print("\nПорада: для стабільних транзакцій бажано мати 30-50 TRX.")


if __name__ == "__main__":
    check_balance()
