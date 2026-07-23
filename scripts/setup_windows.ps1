# Amalfi — one-shot Windows setup for a laptop.
# Installs Git + Python + Visual Studio C++ Build Tools, clones the repo if needed,
# then builds llama.cpp and probes this machine. Idempotent — safe to re-run.
#
# RUN IN AN ELEVATED (Administrator) PowerShell. Two ways:
#
# The LAN IP is auto-detected (and printed). Pass -RpcHost <ip> only to override it.
#
#   A) already cloned the repo:
#        powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#
#   B) fresh laptop, one-liner (installs git, clones, sets up):
#        & ([scriptblock]::Create((irm https://raw.githubusercontent.com/shahinmv/project-amalfi/main/scripts/setup_windows.ps1)))
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
function Test-HasCommand($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }
function Test-HasPython {
  if (-not (Test-HasCommand python)) { return $false }
  try { return ((python --version 2>&1) -match 'Python 3\.') } catch { return $false }
}
function Test-HasVCTools {
  # The C++ toolset specifically (not just "some VS installed").
  $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vsw)) { return $false }
  $p = & $vsw -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null
  return [bool]$p
}

if (-not (Test-Admin)) {
  Write-Host "Must run as Administrator (to install the C++ Build Tools)." -ForegroundColor Red
  Write-Host "Right-click PowerShell -> 'Run as administrator', then re-run this command." -ForegroundColor Yellow
  exit 1
}
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
  throw "winget not found. Install 'App Installer' from the Microsoft Store, then re-run."
}
# Allow the child .ps1 files (bootstrap/build/worker) to run in THIS process even if the
# machine's default execution policy is Restricted. Process scope only — nothing persists.
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
# GitHub needs TLS 1.2 on older PowerShell 5.1 setups.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$wg = @("--accept-source-agreements","--accept-package-agreements","--silent")

Write-Host "== [1/6] Git ==" -ForegroundColor Cyan
if (Test-HasCommand git) { Write-Host ">> already installed, skipping" }
else { winget install --id Git.Git -e @wg 2>$null | Out-Null; Refresh-Path }

Write-Host "== [2/6] Python 3.12 ==" -ForegroundColor Cyan
if (Test-HasPython) { Write-Host ">> already installed, skipping" }
else { winget install --id Python.Python.3.12 -e @wg 2>$null | Out-Null; Refresh-Path }

Write-Host "== [3/6] Visual Studio C++ Build Tools ==" -ForegroundColor Cyan
if (Test-HasVCTools) { Write-Host ">> already installed, skipping" }
else {
  Write-Host ">> installing (large download, several minutes, please wait)..."
  winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-source-agreements --accept-package-agreements `
    --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" 2>$null | Out-Null
}

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
