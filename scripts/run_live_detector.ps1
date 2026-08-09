# Resolve project root path
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $ProjectRoot

# Set PYTHONPATH environment variable to project root
$env:PYTHONPATH = $ProjectRoot.Path

Write-Host "Starting Live Stream Cat Detection..." -ForegroundColor Green
python src/live_detector/live_detector.py