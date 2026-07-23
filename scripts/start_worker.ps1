param([int]$Port = 50052)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
$Bin = "$Here/vendor/llama.cpp/build/bin/rpc-server.exe"
if (-not (Test-Path $Bin)) { throw "$Bin not built. Run build_llamacpp.ps1 first." }
Write-Host ">> starting rpc-server on 0.0.0.0:$Port (LAN-only)"
& $Bin --host 0.0.0.0 --port $Port
