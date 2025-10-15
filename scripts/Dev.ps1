param(
    [Parameter(Position = 0)]
    [string]
    $Task = "help"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

$VenvPath = Join-Path $RepoRoot ".venv"
$ActivateScript = Join-Path $VenvPath "Scripts/Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    $ActivateScript = $null
}
$ACT = $ActivateScript

function Ensure-Venv {
    if (-not (Test-Path $VenvPath)) {
        python -m venv $VenvPath | Out-Null
    }
    if (-not $ACT) {
        $script = Join-Path $VenvPath "Scripts/Activate.ps1"
        if (Test-Path $script) {
            Set-Variable -Name ACT -Value $script -Scope Script
        }
    }
}

function Setup {
    Ensure-Venv
    if ($ACT) {
        . $ACT
    }
    pip install -r requirements.txt
}

function Run-Api {
    Ensure-Venv
    if ($ACT) {
        . $ACT
    }
    uvicorn src.api.app:app --reload --port 8000
}

function Run-Ui {
    Ensure-Venv
    if ($ACT) {
        . $ACT
    }
    streamlit run src/ui/app_streamlit.py
}

function Run-Both {
    Ensure-Venv
    $activate = $ACT
    $apiJob = Start-Job -ScriptBlock {
        param($Root, $Activate)
        Set-Location $Root
        if ($Activate) {
            . $Activate
        }
        uvicorn src.api.app:app --reload --port 8000
    } -ArgumentList $RepoRoot, $activate
    try {
        Start-Sleep -Seconds 2
        if ($activate) {
            . $activate
        }
        streamlit run src/ui/app_streamlit.py
    } finally {
        if ($apiJob) {
            Stop-Job $apiJob -ErrorAction SilentlyContinue
            Remove-Job $apiJob -Force -ErrorAction SilentlyContinue
        }
    }
}

function Hist-Snapshot {
  if ($ACT) {
    . $ACT
  }
  python - <<'PY'
from historian.export import summarize
import os
p = os.getenv('HIST_LEDGER', 'data/historian/ledger.jsonl')
print(summarize(p) if os.path.exists(p) else {"message": "no ledger"})
PY
}

function Hist-Clear {
  $p = $env:HIST_LEDGER
  if (-not $p) { $p = "data/historian/ledger.jsonl" }
  if (Test-Path $p) {
    Remove-Item $p
    "Cleared $p"
  } else {
    "No ledger at $p"
  }
}

switch ($Task.ToLowerInvariant()) {
  'setup'         { Setup; break }
  'run-api'       { Run-Api; break }
  'run-ui'        { Run-Ui; break }
  'run-both'      { Run-Both; break }
  'hist-snapshot' { Hist-Snapshot; break }
  'hist-clear'    { Hist-Clear; break }
  default {
    "Unknown task '$Task'. Available: setup, run-api, run-ui, run-both, hist-snapshot, hist-clear."
  }
}
