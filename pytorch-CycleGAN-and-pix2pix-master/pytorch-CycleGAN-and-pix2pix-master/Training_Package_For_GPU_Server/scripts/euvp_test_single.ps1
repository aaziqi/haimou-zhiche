# PowerShell script to run single-direction generator on EUVP validation or any folder
# This uses --model test and --dataset_mode single
# Example:
#   ./scripts/euvp_test_single.ps1 -DataRoot "d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp" -CheckpointsName "euvp_cyclegan" -Epoch "latest" -Direction "AtoB"

param(
    [string]$DataRoot = "d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp",
    [string]$CheckpointsName = "euvp_cyclegan",
    [string]$Epoch = "latest",
    [string]$Direction = "AtoB",  # AtoB: netG maps A->B, BtoA: netG maps B->A
    [string]$GpuIds = "0",
    [string]$NetG = "resnet_9blocks",
    [string]$Norm = "instance",
    [string]$ResultsDir = "./results/",
    [int]$NumTest = 20
)

$repoTest = Join-Path $PSScriptRoot "..\test.py"

# Detect available checkpoint files and set model_suffix accordingly
$CkptRoot = Join-Path $PSScriptRoot "..\checkpoints"
$CkptDir = Join-Path $CkptRoot $CheckpointsName
$LatestA = Join-Path $CkptDir ("{0}_net_G_A.pth" -f $Epoch)
$LatestB = Join-Path $CkptDir ("{0}_net_G_B.pth" -f $Epoch)
$LatestG = Join-Path $CkptDir ("{0}_net_G.pth" -f $Epoch)

$hasA = Test-Path $LatestA
$hasB = Test-Path $LatestB
$hasG = Test-Path $LatestG

if ($Direction -eq "BtoA") {
  if ($hasB) {
    $ModelSuffix = "_B"
    Write-Host "Found net_G_B: using model_suffix _B (BtoA)."
  } elseif ($hasA) {
    $ModelSuffix = "_A"
    Write-Host "Found net_G_A: using model_suffix _A (fallback for BtoA)."
  } elseif ($hasG) {
    $ModelSuffix = ""
    Write-Host "Found single net_G: using no model_suffix."
  } else {
    $ModelSuffix = $null
  }
} else {
  if ($hasA) {
    $ModelSuffix = "_A"
    Write-Host "Found net_G_A: using model_suffix _A (AtoB)."
  } elseif ($hasB) {
    $ModelSuffix = "_B"
    Write-Host "Found net_G_B: using model_suffix _B (fallback for AtoB)."
  } elseif ($hasG) {
    $ModelSuffix = ""
    Write-Host "Found single net_G: using no model_suffix."
  } else {
    $ModelSuffix = $null
  }
}

if ($null -eq $ModelSuffix) {
  Write-Host "No checkpoints found for '$CheckpointsName' in $CkptDir."
  Write-Host "Ensure the folder exists and contains '$Epoch'_net_G(.pth|_A.pth|_B.pth)."
  exit 1
}

if ($ModelSuffix -eq "") {
  $ModelSuffix = ""
  Write-Host "Found single net_G: using no model_suffix."
}

if ($ModelSuffix -ne "") {
  python "$repoTest" `
    --dataroot "$DataRoot" `
    --name "$CheckpointsName" `
    --model test `
    --dataset_mode single `
    --netG "$NetG" `
    --norm "$Norm" `
    --results_dir "$ResultsDir" `
    --model_suffix "$ModelSuffix" `
    --epoch "$Epoch" `
    --num_test $NumTest
} else {
  python "$repoTest" `
    --dataroot "$DataRoot" `
    --name "$CheckpointsName" `
    --model test `
    --dataset_mode single `
    --netG "$NetG" `
    --norm "$Norm" `
    --results_dir "$ResultsDir" `
    --epoch "$Epoch" `
    --num_test $NumTest
}
