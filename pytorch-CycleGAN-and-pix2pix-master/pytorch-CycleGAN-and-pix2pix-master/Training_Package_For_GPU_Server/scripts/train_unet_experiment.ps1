# PowerShell script to train CycleGAN with U-Net generator on EUVP Unpaired dataset
# Usage: run from anywhere: ./scripts/train_unet_experiment.ps1

param(
    [string]$DataRoot = "d:\VScode\Graduation project\EUVP_Unpaired",
    [string]$Name = "euvp_cyclegan_unet",
    [string]$GpuIds = "0",
    [int]$Seed = 0,
    [int]$NEpochs = 100,
    [int]$NEpochsDecay = 100,
    [float]$LambdaGray = 0.0,
    [float]$LambdaStruct = 0.0,
    [float]$LambdaPerceptual = 0.0,
    [float]$LambdaColor = 0.0,
    [float]$LambdaTv = 0.0,
    [int]$MaxDatasetSize = 0,
    [int]$DisplayFreq = 1000,
    [int]$UpdateHtmlFreq = 1000,
    [int]$PrintFreq = 100,
    [int]$SaveLatestFreq = 5000,
    [int]$SaveEpochFreq = 10,
    [switch]$NoHtml
)

$repoTrain = Join-Path $PSScriptRoot "..\train.py"
# Use the venv python
$pythonExe = "D:\VScode\Graduation project\.venv\Scripts\python.exe"

# 通过 CUDA_VISIBLE_DEVICES 控制设备；"-1" 表示 CPU
if ($GpuIds -eq "-1") {
  $env:CUDA_VISIBLE_DEVICES = ""
} else {
  $env:CUDA_VISIBLE_DEVICES = $GpuIds
}

$argsList = @(
  "--dataroot", "$DataRoot",
  "--name", "$Name",
  "--model", "cycle_gan",
  "--netG", "unet_256",
  "--pool_size", "50",
  "--no_dropout",
  "--seed", "$Seed",
  "--n_epochs", "$NEpochs",
  "--n_epochs_decay", "$NEpochsDecay",
  "--lambda_gray", "$LambdaGray",
  "--lambda_struct", "$LambdaStruct",
  "--lambda_perceptual", "$LambdaPerceptual",
  "--lambda_color", "$LambdaColor",
  "--lambda_tv", "$LambdaTv",
  "--display_freq", "$DisplayFreq",
  "--update_html_freq", "$UpdateHtmlFreq",
  "--print_freq", "$PrintFreq",
  "--save_latest_freq", "$SaveLatestFreq",
  "--save_epoch_freq", "$SaveEpochFreq"
)

if ($MaxDatasetSize -gt 0) {
  $argsList += "--max_dataset_size", "$MaxDatasetSize"
}

if ($NoHtml) {
  $argsList += "--no_html"
}

Write-Host "Starting training with U-Net generator..."
Write-Host "Command: $pythonExe $repoTrain $argsList"

& $pythonExe "$repoTrain" @argsList
if ($LASTEXITCODE -ne 0) {
  Write-Error "Training failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

Write-Host "Training finished successfully."
