param([int]$Port = 50052)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
# VS (multi-config) puts exes in build/bin/Release; Ninja/Make in build/bin.
$Candidates = @(
  "$Here/vendor/llama.cpp/build/bin/ggml-rpc-server.exe",
  "$Here/vendor/llama.cpp/build/bin/Release/ggml-rpc-server.exe"
)
$Bin = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Bin) {
  throw "ggml-rpc-server.exe not found in build/bin or build/bin/Release. Run build_llamacpp.ps1 first."
}
Write-Host ">> starting ggml-rpc-server on 0.0.0.0:$Port (LAN-only)"
& $Bin --host 0.0.0.0 --port $Port
