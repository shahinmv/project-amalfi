# Build a pinned llama.cpp (rpc-server + llama-server) for a given backend.
# Usage: scripts/build_llamacpp.ps1 -Backend cuda|vulkan|cpu
param([Parameter(Mandatory=$true)][ValidateSet("cuda","vulkan","cpu")][string]$Backend)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
Get-Content "$Here/config/llamacpp.pin" | Where-Object { $_ -match "^\w+=" } | ForEach-Object {
  $k,$v = $_ -split "=",2; Set-Variable -Name $k -Value $v
}
Write-Host ">> pinned tag: $LLAMACPP_REF"
$Src = "$Here/vendor/llama.cpp"
if (-not (Test-Path $Src)) {
  git clone --depth 1 --branch $LLAMACPP_REF $LLAMACPP_REPO $Src
}
$Flags = (python "$Here/scripts/build_flags.py" $Backend) -split " "
cmake -S $Src -B "$Src/build" @Flags
cmake --build "$Src/build" --config Release --target ggml-rpc-server llama-server
Write-Host ">> built ggml-rpc-server + llama-server in $Src/build/bin"
