# Build a pinned llama.cpp (rpc-server + llama-server) for a given backend.
# Usage: scripts/build_llamacpp.ps1 -Backend cuda|vulkan|cpu
# Requires Visual Studio Build Tools 2022 with the "Desktop development with C++" workload.
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

# Drop a stale build dir if it was configured with a different generator
# (e.g. a prior failed NMake attempt) — CMake refuses to switch generators in place.
$cache = "$Src/build/CMakeCache.txt"
if (Test-Path $cache) {
  $genLine = Select-String -Path $cache -Pattern "^CMAKE_GENERATOR:INTERNAL=(.*)$"
  $gen = if ($genLine) { $genLine.Matches[0].Groups[1].Value } else { "" }
  if ($gen -ne "Visual Studio 17 2022") {
    Write-Host ">> removing stale build dir (previous generator: '$gen')"
    Remove-Item -Recurse -Force "$Src/build"
  }
}

# Force the Visual Studio generator so CMake finds MSVC automatically (no Developer
# prompt needed) — avoids the "NMake Makefiles / CMAKE_C_COMPILER not set" fallback.
cmake -S $Src -B "$Src/build" -G "Visual Studio 17 2022" -A x64 @Flags
if ($LASTEXITCODE -ne 0) {
  throw "CMake configure failed. Ensure VS Build Tools 2022 with 'Desktop development with C++' is installed (see docs/runbook.md)."
}
cmake --build "$Src/build" --config Release --target ggml-rpc-server llama-server
if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }

$exe = Get-ChildItem -Path "$Src/build" -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exe) { throw "Build reported done but llama-server.exe not found under $Src/build." }
Write-Host ">> built ggml-rpc-server + llama-server in $($exe.DirectoryName)"
