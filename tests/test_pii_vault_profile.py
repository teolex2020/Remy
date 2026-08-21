from types import SimpleNamespace
from unittest.mock import patch

from remy.core.pii_vault import PIIVault


def test_pii_vault_load_profile_ignores_none_metadata_values():
    vault = PIIVault()
    profile = SimpleNamespace(
        metadata={
            "name": None,
            "phone": None,
            "email": "alice@example.com",
            "family": ["Anna", None, " Petro "],
        }
    )
    person = SimpleNamespace(metadata={"name": None})
    brain = SimpleNamespace(search=lambda **kwargs: [person])

    with patch("remy.core.agent_tools.brain", brain), \
         patch("remy.core.agent_tools.brain_lock"), \
         patch("remy.core.brain_tools.get_user_profile_record", return_value=profile):
        vault.load_profile()

    assert "alice@example.com" in vault._profile_values
    assert "Anna" in vault._profile_values
    assert "Petro" in vault._profile_values
