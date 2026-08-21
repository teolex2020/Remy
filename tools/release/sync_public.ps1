[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Destination,
    [string]$Source = "",
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Source) {
    $Source = Join-Path $ScriptDir "..\.."
}

$SourceRoot = [System.IO.Path]::GetFullPath($Source).TrimEnd('\')
$DestinationRoot = [System.IO.Path]::GetFullPath($Destination).TrimEnd('\')

if ($SourceRoot.Equals($DestinationRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Source and destination must be different repositories."
}
if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot "pyproject.toml"))) {
    throw "Source does not look like the Remy project: $SourceRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $DestinationRoot ".git"))) {
    throw "Destination is not a Git working tree: $DestinationRoot"
}

$MirrorDirectories = @(
    "src\remy",
    "tests"
)
$MergeDirectories = @(
    "packaging",
    ".github\workflows",
    "benchmarks",
    "data\evals\llm_optimization",
    "docs",
    "scripts",
    "tools\maintenance",
    "tools\release",
    "tools\validation"
)
$RootFiles = @(
    ".env.example",
    ".gitignore",
    "LICENSE",
    "README.md",
    "pyproject.toml"
)

function Convert-ToRelativePath {
    param([string]$Root, [string]$FullPath)

    $rootUri = [System.Uri]::new(($Root.TrimEnd('\') + '\'))
    $pathUri = [System.Uri]::new($FullPath)
    return [System.Uri]::UnescapeDataString(
        $rootUri.MakeRelativeUri($pathUri).ToString()
    ).Replace('/', '\')
}

function Test-ExcludedPath {
    param([string]$RelativePath)

    $normalized = $RelativePath.Replace('\', '/').TrimStart('/')
    $segments = @($normalized.Split('/'))
    $leaf = $segments[-1]
    $isPublicEvalFixture = $normalized.StartsWith(
        "data/evals/llm_optimization/",
        [System.StringComparison]::OrdinalIgnoreCase
    )

    if ((-not $isPublicEvalFixture) -and ($segments | Where-Object {
        $_ -in @(
            ".build", ".pytest_cache", ".ruff_cache", ".tmp", ".venv",
            "__pycache__", "data", "MagicMock", "node_modules",
            "secrets", "test_verify", "vendor"
        ) -or $_ -like "test-tmp-*" -or $_ -like ".pytest_tmp*" -or $_ -like ".tmp_pytest*"
    })) {
        return $true
    }

    # Exclude the repository's generated release output while still allowing
    # the checked-in exporter under tools/release.
    if ($normalized -eq "release" -or $normalized.StartsWith("release/")) {
        return $true
    }

    if ($leaf -in @(
        ".env", "credentials.json", "desktop.ini", "generated_password.txt",
        "secrets.json", "Thumbs.db"
    )) {
        return $true
    }
    if ($leaf -like ".env.*" -and $leaf -ne ".env.example") {
        return $true
    }
    if ($leaf -match '\.(?:bak|backup|cache|key|log|pem|pyc|pyo|tmp|whl)$') {
        return $true
    }
    return $false
}

function Get-SafeFiles {
    param([string]$Root, [string]$RelativeDirectory)

    $directory = Join-Path $Root $RelativeDirectory
    if (-not (Test-Path -LiteralPath $directory)) {
        return @()
    }

    return @(
        Get-ChildItem -LiteralPath $directory -Recurse -File -Force |
            Where-Object { -not ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) } |
            ForEach-Object {
                $relative = Convert-ToRelativePath -Root $Root -FullPath $_.FullName
                if (-not (Test-ExcludedPath $relative)) {
                    [pscustomobject]@{
                        RelativePath = $relative
                        FullPath = $_.FullName
                    }
                }
            }
    )
}

function Test-SameFile {
    param([string]$Left, [string]$Right)

    if (-not (Test-Path -LiteralPath $Right)) {
        return $false
    }
    $leftItem = Get-Item -LiteralPath $Left
    $rightItem = Get-Item -LiteralPath $Right
    if ($leftItem.Length -ne $rightItem.Length) {
        return $false
    }
    return (Get-FileHash -LiteralPath $Left -Algorithm SHA256).Hash -eq
        (Get-FileHash -LiteralPath $Right -Algorithm SHA256).Hash
}

$operations = [System.Collections.Generic.List[object]]::new()

foreach ($relativeFile in $RootFiles) {
    $sourceFile = Join-Path $SourceRoot $relativeFile
    if (Test-Path -LiteralPath $sourceFile) {
        $destinationFile = Join-Path $DestinationRoot $relativeFile
        if (-not (Test-SameFile -Left $sourceFile -Right $destinationFile)) {
            $operations.Add([pscustomobject]@{
                Action = "COPY"
                RelativePath = $relativeFile
                SourcePath = $sourceFile
            })
        }
    }
}

foreach ($relativeDirectory in @($MirrorDirectories + $MergeDirectories)) {
    foreach ($sourceFile in Get-SafeFiles -Root $SourceRoot -RelativeDirectory $relativeDirectory) {
        $destinationFile = Join-Path $DestinationRoot $sourceFile.RelativePath
        if (-not (Test-SameFile -Left $sourceFile.FullPath -Right $destinationFile)) {
            $operations.Add([pscustomobject]@{
                Action = "COPY"
                RelativePath = $sourceFile.RelativePath
                SourcePath = $sourceFile.FullPath
            })
        }
    }
}

foreach ($relativeDirectory in $MirrorDirectories) {
    $sourcePaths = @{}
    foreach ($sourceFile in Get-SafeFiles -Root $SourceRoot -RelativeDirectory $relativeDirectory) {
        $sourcePaths[$sourceFile.RelativePath.ToLowerInvariant()] = $true
    }
    foreach ($destinationFile in Get-SafeFiles -Root $DestinationRoot -RelativeDirectory $relativeDirectory) {
        if (-not $sourcePaths.ContainsKey($destinationFile.RelativePath.ToLowerInvariant())) {
            $operations.Add([pscustomobject]@{
                Action = "REMOVE"
                RelativePath = $destinationFile.RelativePath
                SourcePath = ""
            })
        }
    }
}

$operations = @($operations | Sort-Object RelativePath, Action)
$copyCount = @($operations | Where-Object Action -eq "COPY").Count
$removeCount = @($operations | Where-Object Action -eq "REMOVE").Count

Write-Host "Remy public sync"
Write-Host "  Source:      $SourceRoot"
Write-Host "  Destination: $DestinationRoot"
Write-Host "  Copy/update: $copyCount"
Write-Host "  Remove:      $removeCount"
Write-Host "  Mode:        $(if ($Apply) { 'APPLY' } else { 'DRY RUN' })"

foreach ($operation in $operations) {
    Write-Output ("{0}`t{1}" -f $operation.Action, $operation.RelativePath)
}

if (-not $Apply) {
    Write-Host "Dry run only. Re-run with -Apply after reviewing the plan."
    return
}

$destinationPrefix = $DestinationRoot.TrimEnd('\') + '\'
foreach ($operation in $operations) {
    $target = [System.IO.Path]::GetFullPath(
        (Join-Path $DestinationRoot $operation.RelativePath)
    )
    if (-not $target.StartsWith(
        $destinationPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing operation outside destination: $target"
    }

    if ($operation.Action -eq "COPY") {
        $parent = Split-Path -Parent $target
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        Copy-Item -LiteralPath $operation.SourcePath -Destination $target -Force
    }
    elseif ($operation.Action -eq "REMOVE" -and (Test-Path -LiteralPath $target)) {
        Remove-Item -LiteralPath $target -Force
    }
}

Write-Host "Public sync applied successfully."
