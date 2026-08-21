[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Version = "",
    [switch]$SkipBrowser,
    [switch]$SkipInstaller,
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "The Remy Windows installer must be built on Windows."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$BuildRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot ".build\windows"))
$ReleaseDir = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "release"))
$DistRoot = Join-Path $BuildRoot "dist"
$WorkRoot = Join-Path $BuildRoot "work"
$VenvDir = Join-Path $BuildRoot ".venv"

if (-not $BuildRoot.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use a build directory outside the repository: $BuildRoot"
}

if ($Clean -and (Test-Path -LiteralPath $BuildRoot)) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $ReleaseDir | Out-Null

Push-Location $RepoRoot
try {
    if (-not $Version) {
        $Version = (& $Python -c "import pathlib,tomllib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))['project']['version'])").Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Could not read the project version with $Python."
        }
    }

    if ($Version -notmatch '^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$') {
        throw "Version must use semantic version syntax, for example 0.9.0. Received: $Version"
    }

    if (-not (Test-Path -LiteralPath (Join-Path $VenvDir "Scripts\python.exe"))) {
        & $Python -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the isolated packaging environment."
        }
    }

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $PyInstaller = Join-Path $VenvDir "Scripts\pyinstaller.exe"

    & $VenvPython -m pip install --disable-pip-version-check --timeout 30 --retries 2 --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "Failed to install build tooling." }

    & $VenvPython -m pip install --disable-pip-version-check --timeout 30 --retries 2 --prefer-binary -e "${RepoRoot}[all]" "pyinstaller>=6.10,<7"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install Remy packaging dependencies." }

    $env:PLAYWRIGHT_BROWSERS_PATH = "0"
    if (-not $SkipBrowser) {
        & $VenvPython -m playwright install chromium
        if ($LASTEXITCODE -ne 0) { throw "Failed to download the bundled Chromium runtime." }
    }

    $NumericVersion = ($Version -split '[-+]', 2)[0]
    $VersionParts = @($NumericVersion.Split('.'))
    $FileVersion = "$($VersionParts[0]), $($VersionParts[1]), $($VersionParts[2]), 0"
    $VersionFile = Join-Path $BuildRoot "version_info.txt"
    $VersionInfo = @"
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=($FileVersion),
    prodvers=($FileVersion),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'AuroraSeed'),
        StringStruct('FileDescription', 'Remy desktop agent'),
        StringStruct('FileVersion', '$Version'),
        StringStruct('InternalName', 'Remy'),
        StringStruct('OriginalFilename', 'Remy.exe'),
        StringStruct('ProductName', 'Remy'),
        StringStruct('ProductVersion', '$Version')
      ])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"@
    Set-Content -LiteralPath $VersionFile -Value $VersionInfo -Encoding UTF8
    $env:REMY_VERSION_FILE = $VersionFile

    & $PyInstaller (Join-Path $ScriptDir "remy.spec") --noconfirm --clean --distpath $DistRoot --workpath $WorkRoot
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

    $RemyExe = Join-Path $DistRoot "Remy\Remy.exe"
    if (-not (Test-Path -LiteralPath $RemyExe)) {
        throw "Standalone build did not produce $RemyExe"
    }

    if ($SkipInstaller) {
        Write-Host "Standalone Remy build: $RemyExe"
        return
    }

    $IsccCommand = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    $IsccCandidates = @(
        if ($IsccCommand) { $IsccCommand.Source }
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    $Iscc = $IsccCandidates | Select-Object -First 1
    if (-not $Iscc) {
        throw "Inno Setup 6 was not found. Install it, then run this script again."
    }

    & $Iscc "/DAppVersion=$Version" "/DNumericVersion=$NumericVersion.0" "/DSourceDir=$($DistRoot)\Remy" "/DOutputDir=$ReleaseDir" (Join-Path $ScriptDir "Remy.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup compiler failed." }

    $SetupExe = Join-Path $ReleaseDir "RemySetup.exe"
    if (-not (Test-Path -LiteralPath $SetupExe)) {
        throw "Installer did not produce $SetupExe"
    }

    $Hash = (Get-FileHash -LiteralPath $SetupExe -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath (Join-Path $ReleaseDir "RemySetup.sha256") -Value "$Hash  RemySetup.exe" -Encoding ASCII

    Write-Host "Remy installer: $SetupExe"
    Write-Host "SHA256: $Hash"
}
finally {
    Pop-Location
}
