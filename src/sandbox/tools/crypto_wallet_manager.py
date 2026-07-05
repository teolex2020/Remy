import eth_account
from eth_account import Account
import eth_account.messages

TOOL_NAME = "crypto_wallet_manager"
TOOL_DESCRIPTION = "Manages EVM-compatible cryptocurrency wallets for autonomous transactions. Supports creation and signing."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["create", "get_address", "sign_message"], "description": "Action to perform"},
        "private_key": {"type": "string", "description": "Private key for signing (optional for 'create')"},
        "message": {"type": "string", "description": "Message to sign (required for 'sign_message')"}
    },
    "required": ["action"]
}

def execute(action, private_key=None, message=None):
    if action == "create":
        acct = Account.create()
        return {"address": acct.address, "private_key": acct.key.hex()}
    elif action == "get_address":
        if not private_key: return {"error": "Private key required"}
        try:
            acct = Account.from_key(private_key)
            return {"address": acct.address}
        except Exception as e:
            return {"error": str(e)}
    elif action == "sign_message":
        if not private_key or not message: return {"error": "Private key and message required"}
        try:
            acct = Account.from_key(private_key)
            signed = acct.sign_message(eth_account.messages.encode_defunct(text=message))
            return {"signature": signed.signature.hex(), "address": acct.address}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Invalid action"}

def test_create():
    result = execute("create")
    assert "address" in result
    assert "private_key" in result
    assert result["address"].startswith("0x")

def test_sign():
    acct = Account.create()
    msg = "Hello World"
    result = execute("sign_message", private_key=acct.key.hex(), message=msg)
    assert "signature" in result
    assert result["address"] == acct.address
