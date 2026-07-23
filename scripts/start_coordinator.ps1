param([string]$Fleet = "")
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
if ($Fleet -eq "") { $Fleet = "$Here/fleet.json" }
if (-not (Test-Path $Fleet)) { throw "$Fleet not found. Run plan_split.py first." }
$Cmd = (python -c "import json;print(json.load(open(r'$Fleet'))['coordinator_cmd'])")
# VS (multi-config) puts exes in build/bin/Release; Ninja/Make in build/bin.
$BinDirs = @("$Here/vendor/llama.cpp/build/bin/Release", "$Here/vendor/llama.cpp/build/bin")
$BinDir = $BinDirs | Where-Object { Test-Path "$_/llama-server.exe" } | Select-Object -First 1
if (-not $BinDir) {
  throw "llama-server.exe not found in build/bin or build/bin/Release. Run build_llamacpp.ps1 first."
}
$env:PATH = "$BinDir;$env:PATH"
Write-Host ">> $Cmd"
Set-Location $Here
Invoke-Expression $Cmd
