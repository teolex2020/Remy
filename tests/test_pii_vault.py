from remy.core.pii_vault import PIIVault, StreamingRestorer


def test_pii_vault_restores_model_mutated_name_token_alias():
    vault = PIIVault()
    token = vault.tokenize("Alice", "name")

    assert token == "[PII:name_1]"
    assert vault.restore_text("Hello [PII:name_1]") == "Hello Alice"
    assert vault.restore_text("Hello [PI!name_1]") == "Hello Alice"
    assert vault.restore_text("Hello [PII!name_1]") == "Hello Alice"


def test_streaming_restorer_buffers_split_pii_tokens_before_display():
    vault = PIIVault()
    vault.tokenize("Alice", "name")
    restorer = StreamingRestorer(vault)

    assert restorer.feed("Hello [PI") == "Hello "
    assert restorer.feed("I:name_1], welcome") == "Alice, welcome"
    assert restorer.flush() == ""


def test_streaming_restorer_restores_split_mutated_pii_token_alias():
    vault = PIIVault()
    vault.tokenize("Alice", "name")
    restorer = StreamingRestorer(vault)

    assert restorer.feed("Hello [PI") == "Hello "
    assert restorer.feed("!name_1], welcome") == "Alice, welcome"
    assert restorer.flush() == ""
