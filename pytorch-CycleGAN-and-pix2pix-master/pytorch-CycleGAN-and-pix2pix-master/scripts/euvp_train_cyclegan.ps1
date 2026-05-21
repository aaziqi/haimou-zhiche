# PowerShell script to train CycleGAN on EUVP Unpaired dataset
# Usage: run from anywhere: ./scripts/euvp_train_cyclegan.ps1

param(
    [string]$DataRoot = "d:\VScode\Graduation project\EUVP_Unpaired",
    [string]$Name = "euvp_cyclegan",
    [string]$GpuIds = "0",
    [int]$Seed = 0,
    [int]$NEpochs = 100,
    [int]$NEpochsDecay = 100,
    [float]$LambdaGray = 0.0,
    [float]$LambdaStruct = 0.0,
    [float]$LambdaPerceptual = 0.0,
    [float]$LambdaColor = 0.0,
    [int]$MaxDatasetSize = 0,
    [int]$DisplayFreq = 1000000000,
    [int]$UpdateHtmlFreq = 1000000000,
    [int]$PrintFreq = 100,
    [int]$SaveLatestFreq = 5000,
    [int]$SaveEpochFreq = 10,
    [switch]$NoHtml,
    [switch]$UseWandb,
    [switch]$AutoBenchmark,
    [string]$BenchmarkDataset = "euvp",
    [string]$TestInp = "d:\VScode\Graduation project\EUVP Dataset\test_samples\Inp",
    [string]$TestRef = "d:\VScode\Graduation project\EUVP Dataset\test_samples\GTr",
    [string]$TestDirection = "AtoB",
    [int]$NumTest = 200,
    [int]$BootstrapIters = 2000,
    [int]$BenchmarkSeed = 123,
    [switch]$ReportUIQMParts
)

$repoTrain = Join-Path $PSScriptRoot "..\train.py"

# 通过 CUDA_VISIBLE_DEVICES 控制设备；"-1" 表示 CPU
if ($GpuIds -eq "-1") {
  $env:CUDA_VISIBLE_DEVICES = ""
} else {
  $env:CUDA_VISIBLE_DEVICES = $GpuIds
}

$args = @(
  "--dataroot", "$DataRoot",
  "--name", "$Name",
  "--model", "cycle_gan",
  "--pool_size", "50",
  "--no_dropout",
  "--seed", "$Seed",
  "--n_epochs", "$NEpochs",
  "--n_epochs_decay", "$NEpochsDecay",
  "--lambda_gray", "$LambdaGray",
  "--lambda_struct", "$LambdaStruct",
  "--lambda_perceptual", "$LambdaPerceptual",
  "--lambda_color", "$LambdaColor",
  "--display_freq", "$DisplayFreq",
  "--update_html_freq", "$UpdateHtmlFreq",
  "--print_freq", "$PrintFreq",
  "--save_latest_freq", "$SaveLatestFreq",
  "--save_epoch_freq", "$SaveEpochFreq"
)

if ($MaxDatasetSize -gt 0) {
  $args += @("--max_dataset_size", "$MaxDatasetSize")
}

if ($NoHtml) {
  $args += @("--no_html")
}

if ($UseWandb) {
  $args += @("--use_wandb")
}

python "$repoTrain" @args
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

if ($AutoBenchmark) {
  $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
  $ckptDir = Join-Path $repoRoot ("checkpoints\{0}" -f $Name)
  $totalEpoch = $NEpochs + $NEpochsDecay
  $epochCandidates = @("$totalEpoch", "latest")
  $evalEpoch = "latest"
  foreach ($e in $epochCandidates) {
    $a = Join-Path $ckptDir ("{0}_net_G_A.pth" -f $e)
    $b = Join-Path $ckptDir ("{0}_net_G_B.pth" -f $e)
    $g = Join-Path $ckptDir ("{0}_net_G.pth" -f $e)
    if ((Test-Path $a) -or (Test-Path $b) -or (Test-Path $g)) {
      $evalEpoch = $e
      break
    }
  }

  & (Join-Path $PSScriptRoot "euvp_test_single.ps1") `
    -DataRoot "$TestInp" `
    -CheckpointsName "$Name" `
    -Epoch "$evalEpoch" `
    -Direction "$TestDirection" `
    -GpuIds "$GpuIds" `
    -NumTest $NumTest
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }

  $predDir = Join-Path $repoRoot ("results\{0}\test_{1}\images" -f $Name, $evalEpoch)
  $perDir = Join-Path $repoRoot ("results\benchmarks\{0}\models\{1}\epoch_{2}" -f $BenchmarkDataset, $Name, $evalEpoch)
  $perCsv = Join-Path $perDir "per_image.csv"
  $perJson = Join-Path $perDir "summary.json"
  $summaryCsv = Join-Path $repoRoot "results\benchmarks\summary.csv"
  $evalScript = Join-Path $repoRoot "scripts\evaluate_euvp_psnr_ssim.py"

  $evalArgs = @(
    "--inp_dir", "$TestInp",
    "--pred_dir", "$predDir",
    "--max_images", "$NumTest",
    "--bootstrap_iters", "$BootstrapIters",
    "--seed", "$BenchmarkSeed",
    "--save_csv", "$perCsv",
    "--save_json", "$perJson",
    "--benchmark_csv", "$summaryCsv",
    "--benchmark_dataset", "$BenchmarkDataset",
    "--benchmark_model", "$Name",
    "--benchmark_epoch", "$evalEpoch"
  )
  if (-not [string]::IsNullOrWhiteSpace($TestRef)) {
    $evalArgs += @("--ref_dir", "$TestRef")
  }
  if ($ReportUIQMParts) {
    $evalArgs += @("--report_uiqm_parts")
  }

  python "$evalScript" @evalArgs
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}
