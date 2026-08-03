<#
.SYNOPSIS
  Start the OCS editor on http://127.0.0.1:8765/
#>
[CmdletBinding()]
param(
  [int]$Port = 8765,
  [string]$Address = "127.0.0.1",
  [switch]$Reload,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "venv missing. Run ./scripts/setup_env.ps1 first."
}

$url = "http://${Address}:${Port}/"
Write-Output "OCS -> $url"
if (-not $NoBrowser) { Start-Process $url }

$args = @("-m", "uvicorn", "ocs.server:app", "--host", $Address, "--port", "$Port")
if ($Reload) { $args += @("--reload", "--reload-dir", (Join-Path $root "ocs")) }

Push-Location $root
try { & $py @args } finally { Pop-Location }
