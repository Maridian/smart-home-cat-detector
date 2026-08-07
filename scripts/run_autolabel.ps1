# Set path to the project root directory
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $ProjectRoot

# Execute virtual environment / Python script
Write-Host "Starting auto-labeling with YOLOv8m..." -ForegroundColor Green
python src/dataset/auto_label.py