<#
.SYNOPSIS
  Download the Spine Web Player runtime into web/vendor/ for offline previews.

.DESCRIPTION
  The Spine Runtimes License Agreement restricts redistribution and requires the
  user to hold their own Spine license, so OCS does not commit this runtime -
  web/vendor/spine-player.* is gitignored and fetched here instead.

  Exported previews inline whatever this script leaves behind, making them work
  from file:// with no network. Skip it and previews fall back to loading the
  runtime from unpkg when opened.

  By running this you are asserting you hold a valid Spine license.
  See http://esotericsoftware.com/spine-runtimes-license
#>
[CmdletBinding()]
param(
  [string]$Version = "4.2.*",
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vendor = Join-Path $root "web\vendor"
New-Item -ItemType Directory -Force -Path $vendor | Out-Null

$targets = @(
  @{ Name = "spine-player.js";  Url = "https://unpkg.com/@esotericsoftware/spine-player@$Version/dist/iife/spine-player.js" }
  @{ Name = "spine-player.css"; Url = "https://unpkg.com/@esotericsoftware/spine-player@$Version/dist/spine-player.css" }
)

foreach ($t in $targets) {
  $dest = Join-Path $vendor $t.Name
  if ((Test-Path $dest) -and -not $Force) {
    $kb = [math]::Round((Get-Item $dest).Length / 1KB)
    Write-Output "skip  $($t.Name) (already present, $kb KB) - use -Force to refresh"
    continue
  }
  Write-Output "fetch $($t.Name) <- $($t.Url)"
  Invoke-WebRequest -Uri $t.Url -OutFile $dest -UseBasicParsing
  $kb = [math]::Round((Get-Item $dest).Length / 1KB)
  Write-Output "      wrote $dest ($kb KB)"
}

Write-Output ""
Write-Output "Done. Previews exported from now on will embed the runtime."
Write-Output "Reminder: using the Spine Runtimes requires your own Spine license."
