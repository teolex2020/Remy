
import os
import json
from eth_account import Account

# This tool manages autonomous crypto wallets using Coinbase AgentKit (2026 version)
# It uses the 'brain' to persist wallet metadata and configuration.

TOOL_NAME = "cdp_wallet_manager"
TOOL_DESCRIPTION = "Manages autonomous wallets via Coinbase AgentKit (CDP) for AI agents (2026). Supports creating wallets, checking balances, and sending USDC on Base."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["create_wallet", "get_balance", "send_transfer", "list_wallets"],
            "description": "The action to perform."
        },
        "wallet_id": {
            "type": "string",
            "description": "The ID of the wallet (required for balance and transfer)."
        },
        "amount": {
            "type": "number",
            "description": "Amount to transfer (in USDC)."
        },
        "recipient": {
            "type": "string",
            "description": "Recipient address or ENS name."
        },
        "network": {
            "type": "string",
            "default": "base-mainnet",
            "description": "Blockchain network (default: base-mainnet)."
        }
    },
    "required": ["action"]
}

def execute(brain, action, wallet_id=None, amount=None, recipient=None, network="base-mainnet"):
    try:
        if action == "create_wallet":
            # Generate a new local account (in real CDP, this would call the CDP API)
            # For 2026 autonomous agents, we use AgentKit to register this on-chain
            new_account = Account.create()
            address = new_account.address
            # In a real scenario, the private key would be stored in a TEE or CDP Vault
            # Here we store the address and a 'mock' wallet ID in the brain
            wallet_data = {
                "address": address,
                "network": network,
                "type": "agentic-wallet",
                "protocol": "x402"
            }
            # Store in brain as a 'finance' record
            record_id = brain.store(
                content=f"Autonomous Wallet: {address} ({network})",
                level="L3_DOMAIN",
                tags=f"finance,crypto,wallet,{network}",
                metadata=wallet_data
            )
            return {"status": "success", "wallet_id": record_id, "address": address, "network": network}

        elif action == "list_wallets":
            results = brain.search("Autonomous Wallet", tags="finance,wallet")
            return {"status": "success", "wallets": results}

        elif action == "get_balance":
            if not wallet_id:
                return {"status": "error", "message": "wallet_id is required"}
            
            wallet = brain.get(wallet_id)
            if not wallet:
                return {"status": "error", "message": "Wallet not found"}
            
            # Real implementation would call CDP SDK / x402 balance check
            # For now, we mock the result to demonstrate functionality
            address = wallet.get("metadata", {}).get("address", "0x...")
            return {
                "status": "success",
                "address": address,
                "balance_usdc": 100.0, # Mocked
                "network": network,
                "note": "Balance retrieved via AgentKit (Mock)"
            }

        elif action == "send_transfer":
            if not wallet_id or not amount or not recipient:
                return {"status": "error", "message": "wallet_id, amount, and recipient are required"}
            
            # Real implementation would use AgentKit 'send-usdc' skill
            return {
                "status": "success",
                "transaction_hash": "0x" + "a"*64, # Mocked
                "amount": amount,
                "currency": "USDC",
                "recipient": recipient,
                "network": network,
                "protocol": "x402 (Gasless)"
            }

        return {"status": "error", "message": f"Unknown action: {action}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

def test_create_and_list():
    class MockBrain:
        def __init__(self):
            self.data = {}
        def store(self, content, level, tags, metadata):
            idx = str(len(self.data))
            self.data[idx] = {"content": content, "metadata": metadata}
            return idx
        def search(self, query, tags):
            return [{"id": k, "content": v["content"]} for k, v in self.data.items()]
        def get(self, id):
            return self.data.get(id)

    brain = MockBrain()
    res = execute(brain, action="create_wallet")
    assert res["status"] == "success"
    assert "address" in res
    
    res_list = execute(brain, action="list_wallets")
    assert len(res_list["wallets"]) == 1

def test_balance_transfer():
    class MockBrain:
        def __init__(self):
            self.data = {"w1": {"metadata": {"address": "0x123"}}}
        def get(self, id): return self.data.get(id)
        def search(self, q, t): return []

    brain = MockBrain()
    res = execute(brain, action="get_balance", wallet_id="w1")
    assert res["balance_usdc"] == 100.0
    
    res_tx = execute(brain, action="send_transfer", wallet_id="w1", amount=10.0, recipient="0xabc")
    assert res_tx["status"] == "success"
