param([string]$Fleet = "")
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
if ($Fleet -eq "") { $Fleet = "$Here/fleet.json" }
if (-not (Test-Path $Fleet)) { throw "$Fleet not found. Run plan_split.py first." }
$Cmd = (python -c "import json;print(json.load(open(r'$Fleet'))['coordinator_cmd'])")
$env:PATH = "$Here/vendor/llama.cpp/build/bin;$env:PATH"
Write-Host ">> $Cmd"
Set-Location $Here
Invoke-Expression $Cmd
