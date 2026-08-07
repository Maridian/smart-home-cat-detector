# Resolve project root path
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $ProjectRoot

# Set PYTHONPATH environment variable to project root
$env:PYTHONPATH = $ProjectRoot.Path

Write-Host "Starting YOLOv8n training..." -ForegroundColor Green
python src/models/trainer.py