<#
.SYNOPSIS
  Create the OCS virtualenv and install everything, including see-through's deps.

.DESCRIPTION
  One shared venv on purpose. see-through's requirements.txt already pins the
  heavy half of what OCS needs (numpy, opencv-python, pillow, scipy,
  scikit-image, psd-tools[composite]), so a second environment would only create
  version conflicts. OCS adds fastapi/uvicorn on top and runs GPU work as a
  subprocess of the same interpreter.

  Verified on: Windows 11, RTX 5070 Ti (sm_120), CUDA 12.8, Python 3.12.13.
  torch 2.8.0+cu128 ships sm_120 kernels, which Blackwell cards require.

.EXAMPLE
  ./scripts/setup_env.ps1
#>
[CmdletBinding()]
param(
  [string]$PythonVersion = "3.12",
  [string]$TorchVersion = "2.8.0",
  [string]$CudaIndex = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$seeThrough = Join-Path $root "external\see-through"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  throw "uv not found. Install it: winget install astral-sh.uv"
}
if (-not (Test-Path (Join-Path $seeThrough "requirements.txt"))) {
  Write-Output "== fetching see-through submodule"
  git -C $root submodule update --init --recursive
}

Write-Output "== creating venv (Python $PythonVersion)"
uv venv --python $PythonVersion $venv
$env:VIRTUAL_ENV = $venv

Write-Output "== installing torch $TorchVersion from $CudaIndex"
uv pip install "torch==$TorchVersion" "torchvision==0.23.0" "torchaudio==$TorchVersion" --index-url $CudaIndex

Write-Output "== installing see-through requirements"
# cwd must be the submodule root: its requirements.txt has relative editable
# entries (-e ./common, -e ./annotators).
Push-Location $seeThrough
try { uv pip install -r requirements.txt } finally { Pop-Location }

Write-Output "== installing OCS web deps"
uv pip install fastapi "uvicorn[standard]" python-multipart pytest

Write-Output "== verifying GPU"
& $py -c @"
import torch
ok = torch.cuda.is_available()
print(f'torch      {torch.__version__}')
print(f'cuda       {ok}')
if ok:
    print(f'device     {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'capability sm_{cap[0]}{cap[1]}')
    print(f'vram       {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB')
    a = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    assert torch.isfinite((a @ a).float()).all()
    print('bf16 matmul OK')
else:
    raise SystemExit('CUDA unavailable - OCS cannot run see-through')
"@

Write-Output ""
Write-Output "Done. Next:"
Write-Output "  ./scripts/fetch_spine_player.ps1   # optional, for offline previews"
Write-Output "  ./scripts/run_ocs.ps1              # start the editor"
