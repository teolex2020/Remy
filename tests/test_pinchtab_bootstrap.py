"""Tests for PinchTab bootstrap/install layer."""

from pathlib import Path
from zipfile import ZipFile

from remy.core import pinchtab_bootstrap as bootstrap


def test_resolve_pinchtab_binary_prefers_explicit_path(tmp_path, monkeypatch):
    binary = tmp_path / ("pinchtab.exe" if bootstrap.os.name == "nt" else "pinchtab")
    binary.write_text("fake-binary", encoding="utf-8")
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_BINARY_PATH", str(binary))
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_COMMAND", "")

    resolved = bootstrap.resolve_pinchtab_binary()

    assert resolved == binary


def test_bootstrap_pinchtab_binary_extracts_local_zip(tmp_path, monkeypatch):
    install_dir = tmp_path / "install"
    archive = tmp_path / "pinchtab.zip"
    binary_name = "pinchtab.exe" if bootstrap.os.name == "nt" else "pinchtab"

    with ZipFile(archive, "w") as zf:
        zf.writestr(f"pinchtab/{binary_name}", "binary-content")

    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_BINARY_PATH", "")
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_COMMAND", "")
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_BOOTSTRAP_SOURCE", str(archive))
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_RELEASE_URL", "")
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_INSTALL_DIR", install_dir)

    result = bootstrap.bootstrap_pinchtab_binary()

    assert result == install_dir / binary_name
    assert result.exists()


def test_build_pinchtab_command_uses_bootstrapped_binary(tmp_path, monkeypatch):
    install_dir = tmp_path / "install"
    install_dir.mkdir(parents=True)
    binary_name = "pinchtab.exe" if bootstrap.os.name == "nt" else "pinchtab"
    binary = install_dir / binary_name
    binary.write_text("fake-binary", encoding="utf-8")

    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_BINARY_PATH", "")
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_COMMAND", "pinchtab serve --host 127.0.0.1 --port 8941")
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_BOOTSTRAP_SOURCE", "")
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_RELEASE_URL", "")
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_INSTALL_DIR", install_dir)
    monkeypatch.setattr(bootstrap.settings, "PINCHTAB_BASE_URL", "http://127.0.0.1:8941")
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: None)

    command = bootstrap.build_pinchtab_command()

    assert command[0] == str(binary)
    assert command[1:] == ["serve", "--host", "127.0.0.1", "--port", "8941"]
