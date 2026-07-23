# One-command per-laptop setup for Amalfi: venv + deps + build llama.cpp + probe.
# Usage:
#   scripts/bootstrap.ps1 [-Backend auto|cuda|vulkan|cpu] [-RpcHost IP] [-RpcPort N] [-StartWorker]
param(
  [string]$Backend = "auto",
  [string]$RpcHost = "",
  [int]$RpcPort = 50052,
  [switch]$StartWorker
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
Set-Location $Here

Write-Host "== [1/4] python venv + deps =="
if (-not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install -q --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -q -r requirements.txt

if ($Backend -eq "auto") {
  $Backend = (& .\.venv\Scripts\python.exe scripts/probe.py --print-backend).Trim()
}
Write-Host "== [2/4] building llama.cpp (backend: $Backend) =="
$env:PATH = "$Here\.venv\Scripts;$env:PATH"
& .\scripts\build_llamacpp.ps1 -Backend $Backend

Write-Host "== [3/4] probing this machine =="
$Out = "node_$($env:COMPUTERNAME).json"
$hostArgs = @()
if ($RpcHost -ne "") { $hostArgs += @("--rpc-host", $RpcHost) }
& .\.venv\Scripts\python.exe scripts/probe.py @hostArgs --rpc-port $RpcPort --out $Out
Write-Host ">> wrote $Out (copy to the coordinator; merge all node_*.json into nodes.json)"

Write-Host "== [4/4] next steps =="
if ($StartWorker) {
  Write-Host ">> starting worker (rpc-server) on port $RpcPort ..."
  & .\scripts\start_worker.ps1 -Port $RpcPort
} else {
  Write-Host "Done. Run on EVERY laptop (add -StartWorker to also launch the worker)."
  Write-Host "On the COORDINATOR: merge node_*.json -> nodes.json, then plan_split.py --model qwen3-30b-a3b-q4,"
  Write-Host "download the GGUF into models\\ (see docs/runbook.md), start workers, then start_coordinator.ps1."
}
