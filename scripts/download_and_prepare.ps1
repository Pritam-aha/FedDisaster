param(
  [string]$DatasetSlug = "saurabhshahane/roadway-flooding-image-dataset",
  [int]$NumClients = 3,
  [double]$ClientTrainRatio = 0.8,
  [double]$GlobalTestRatio = 0.1,
  [switch]$Force
)

# Resolve repo root
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Prepare folders
$dataDownloads = Join-Path $RepoRoot "data\_downloads"
$dataRaw = Join-Path $RepoRoot "data\_raw"
New-Item -ItemType Directory -Force -Path $dataDownloads | Out-Null
New-Item -ItemType Directory -Force -Path $dataRaw | Out-Null

# Check kaggle CLI
$kaggle = Get-Command kaggle -ErrorAction SilentlyContinue
if (-not $kaggle) {
  Write-Host "kaggle CLI not found. Install with: pip install kaggle" -ForegroundColor Yellow
  Write-Host "Ensure your Kaggle API token exists at `$HOME\.kaggle\kaggle.json or set KAGGLE_CONFIG_DIR."
  exit 1
}

# Attempt download & unzip directly to data\_raw
# Note: Requires Kaggle API token configured locally.
$env:KAGGLE_USERNAME = $env:KAGGLE_USERNAME
$env:KAGGLE_KEY = $env:KAGGLE_KEY

Write-Host "Downloading dataset: $DatasetSlug" -ForegroundColor Cyan
kaggle datasets download -d $DatasetSlug -p $dataRaw --unzip
if ($LASTEXITCODE -ne 0) {
  Write-Host "Download failed. Check your Kaggle credentials and slug." -ForegroundColor Red
  exit 1
}

# Run the dataset setup script (auto-detects ImageFolder root under data\_raw)
$forceFlag = $null
if ($Force) { $forceFlag = "--force" }

# Build command arguments
$cmdArgs = @(
  "data\setup_dataset.py",
  "--source_dir", $dataRaw,
  "--target_root", "data",
  "--num_clients", $NumClients,
  "--client_train_ratio", $ClientTrainRatio,
  "--global_test_ratio", $GlobalTestRatio
)

if ($Force) {
  $cmdArgs += "--force"
}

Write-Host "Preparing dataset into clients and global_test..." -ForegroundColor Cyan
& python $cmdArgs
if ($LASTEXITCODE -ne 0) {
  Write-Host "Dataset preparation failed." -ForegroundColor Red
  exit 1
}

Write-Host "Done. Data prepared under .\data\client_* and .\data\global_test" -ForegroundColor Green
