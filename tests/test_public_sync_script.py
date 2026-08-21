"""Safety contracts for the private-to-public release exporter."""

from pathlib import Path


SCRIPT = Path("tools/release/sync_public.ps1")


def test_public_sync_uses_allowlisted_product_surfaces():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"src\\remy"' in source
    assert '"tests"' in source
    assert '"packaging"' in source
    assert '".github\\workflows"' in source
    assert '"docs"' in source
    assert '"data\\evals\\llm_optimization"' in source
    assert '"tools\\release"' in source
    assert '"pyproject.toml"' in source
    assert '$normalized.StartsWith("release/")' in source


def test_public_sync_excludes_private_and_runtime_data():
    source = SCRIPT.read_text(encoding="utf-8")

    for marker in (
        '".env"',
        '"data"',
        '"generated_password.txt"',
        '"secrets.json"',
        '"test-tmp-*"',
        '"__pycache__"',
        '".build"',
    ):
        assert marker in source

    assert '"data/evals/llm_optimization/"' in source


def test_public_sync_is_dry_run_by_default_and_confines_writes():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "[switch]$Apply" in source
    assert "if (-not $Apply)" in source
    assert "$destinationPrefix" in source
    assert "StartsWith(" in source
    assert "Remove-Item -LiteralPath $target -Force" in source
    assert "Remove-Item -LiteralPath $target -Recurse" not in source


def test_public_sync_requires_a_git_destination():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'Join-Path $DestinationRoot ".git"' in source
    assert "Source and destination must be different repositories" in source
