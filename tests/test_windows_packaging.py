"""Static contracts for the reproducible Windows installer pipeline."""

from pathlib import Path


WINDOWS_PACKAGING = Path("packaging/windows")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_pyinstaller_spec_builds_desktop_onedir_with_runtime_assets():
    spec = _read(WINDOWS_PACKAGING / "remy.spec")

    assert 'ENTRYPOINT = PROJECT_ROOT / "src" / "remy" / "desktop_entry.py"' in spec
    assert 'name="Remy"' in spec
    assert "console=False" in spec
    assert "COLLECT(" in spec
    assert 'collect_data_files("remy")' in spec
    assert 'collect_dynamic_libs("aura")' in spec
    assert '"playwright_runtime.py"' in spec


def test_playwright_runtime_uses_bundled_browser_location():
    hook = _read(WINDOWS_PACKAGING / "playwright_runtime.py")

    assert "PLAYWRIGHT_BROWSERS_PATH" in hook
    assert 'getattr(sys, "_MEIPASS"' in hook
    assert ".local-browsers" in hook


def test_inno_setup_is_per_user_and_creates_normal_shortcuts():
    installer = _read(WINDOWS_PACKAGING / "Remy.iss")

    assert "PrivilegesRequired=lowest" in installer
    assert r"DefaultDirName={localappdata}\Programs\Remy" in installer
    assert "OutputBaseFilename=RemySetup" in installer
    assert r'Filename: "{app}\Remy.exe"' in installer
    assert "UninstallDisplayIcon={app}\\Remy.exe" in installer
    assert "runascurrentuser" not in installer.lower()


def test_build_script_uses_isolated_environment_and_all_features():
    script = _read(WINDOWS_PACKAGING / "build.ps1")

    assert "-m venv" in script
    assert '"${RepoRoot}[all]"' in script
    assert "-m playwright install chromium" in script
    assert "pyinstaller>=6.10,<7" in script
    assert 'Get-Command "ISCC.exe"' in script
    assert "RemySetup.sha256" in script
    assert "Remove-Item -LiteralPath $BuildRoot" in script


def test_release_workflow_builds_and_uploads_installer():
    workflow = _read(Path(".github/workflows/windows-installer.yml"))

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" in workflow
    assert "      - main" in workflow
    assert '      - "v*"' in workflow
    assert "runs-on: windows-latest" in workflow
    assert "./packaging/windows/build.ps1" in workflow
    assert "release/RemySetup.exe" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "gh release" in workflow


def test_packaging_never_bundles_local_secrets_file():
    spec = _read(WINDOWS_PACKAGING / "remy.spec")

    assert 'PROJECT_ROOT / ".env"' not in spec
    assert 'PROJECT_ROOT / ".env.example"' in spec
