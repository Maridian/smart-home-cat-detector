# Resolve project root path
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $ProjectRoot

# Set PYTHONPATH environment variable to project root
$env:PYTHONPATH = $ProjectRoot.Path

Write-Host "Starting Model Validation..." -ForegroundColor Green
python src/benchmark/val.py