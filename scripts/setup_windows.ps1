# Amalfi — one-shot Windows setup for a laptop.
# Installs Git + Python + Visual Studio C++ Build Tools, clones the repo if needed,
# then builds llama.cpp and probes this machine. Idempotent — safe to re-run.
#
# RUN IN AN ELEVATED (Administrator) PowerShell. Two ways:
#
#   A) already cloned the repo:
#        powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -RpcHost <LAN-IP>
#
#   B) fresh laptop, one-liner (installs git, clones, sets up):
#        & ([scriptblock]::Create((irm https://raw.githubusercontent.com/shahinmv/project-amalfi/main/scripts/setup_windows.ps1))) -RpcHost <LAN-IP>
#
param(
  [string]$RpcHost = "",
  [ValidateSet("cpu","cuda","vulkan")][string]$Backend = "cpu",
  [int]$RpcPort = 50052,
  [switch]$StartWorker,
  [string]$RepoUrl = "https://github.com/shahinmv/project-amalfi"
)
$ErrorActionPreference = "Stop"

function Test-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  return ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
}
function Refresh-Path {
  $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
              [Environment]::GetEnvironmentVariable("Path","User")
}

if (-not (Test-Admin)) {
  Write-Host "Must run as Administrator (to install the C++ Build Tools)." -ForegroundColor Red
  Write-Host "Right-click PowerShell -> 'Run as administrator', then re-run this command." -ForegroundColor Yellow
  exit 1
}
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
  throw "winget not found. Install 'App Installer' from the Microsoft Store, then re-run."
}

$wg = @("--accept-source-agreements","--accept-package-agreements","--silent")
Write-Host "== [1/6] Git ==" -ForegroundColor Cyan
winget install --id Git.Git -e @wg 2>$null | Out-Null
Write-Host "== [2/6] Python 3.12 ==" -ForegroundColor Cyan
winget install --id Python.Python.3.12 -e @wg 2>$null | Out-Null
Write-Host "== [3/6] Visual Studio C++ Build Tools (large download, please wait) ==" -ForegroundColor Cyan
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-source-agreements --accept-package-agreements `
  --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" 2>$null | Out-Null

Refresh-Path
foreach ($t in @("git","python")) {
  if (-not (Get-Command $t -ErrorAction SilentlyContinue)) {
    throw "$t not on PATH after install. REBOOT the laptop and re-run this script."
  }
}

# Locate the repo: in-repo run, current dir, or clone fresh.
Write-Host "== [4/6] locating repo ==" -ForegroundColor Cyan
$Repo = $null
if ($PSScriptRoot -and (Test-Path (Join-Path (Split-Path $PSScriptRoot -Parent) "requirements.txt"))) {
  $Repo = (Split-Path $PSScriptRoot -Parent)
} elseif (Test-Path ".\requirements.txt") {
  $Repo = (Get-Location).Path
} else {
  $Repo = Join-Path $HOME "Desktop\project-amalfi"
  if (-not (Test-Path (Join-Path $Repo "requirements.txt"))) {
    Write-Host ">> cloning $RepoUrl -> $Repo"
    git clone $RepoUrl $Repo
  }
}
Set-Location $Repo
git pull --ff-only 2>$null | Out-Null
Write-Host ">> repo: $Repo"

Write-Host "== [5/6] build + probe (backend: $Backend) ==" -ForegroundColor Cyan
$btArgs = @("-Backend", $Backend, "-RpcPort", $RpcPort)
if ($RpcHost -ne "") { $btArgs += @("-RpcHost", $RpcHost) }
if ($StartWorker)    { $btArgs += "-StartWorker" }
& "$Repo\scripts\bootstrap.ps1" @btArgs

Write-Host "== [6/6] done ==" -ForegroundColor Green
Write-Host "Copy this machine's node_*.json to the coordinator, then run merge_nodes.py + plan_split.py there."
