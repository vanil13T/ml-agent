param(
    [ValidateSet("regression", "classification")]
    [string]$Task = "regression",
    [int]$Minutes = 20
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path ".venv\.installed")) {
    if (-not (Test-Path $python)) {
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Python 3.12 not found (py -3.12)" }
    }
    & $python -m pip install --default-timeout=300 --retries 10 `
        numpy==1.26.4 scipy==1.12.0 pandas==2.2.3 scikit-learn==1.5.2 `
        lightgbm==4.6.0 catboost==1.2.8 xgboost==2.1.4 `
        openai==1.68.2 httpx==0.28.1 langchain==0.3.27 langchain-openai==0.3.23 langgraph==0.6.7 `
        python-dotenv==1.0.1
    if ($LASTEXITCODE -ne 0) { throw "pip install failed, run again" }
    New-Item ".venv\.installed" -ItemType File | Out-Null
}

$env:TIME_LIMIT_SEC = [string]($Minutes * 60)
& $python solution.py "data\$Task\task.md"
