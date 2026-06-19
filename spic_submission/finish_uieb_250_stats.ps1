$ErrorActionPreference = "Stop"

$projectRoot = "D:\VScode\Graduation project"
$repoRoot = Join-Path $projectRoot "pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master"
$pythonExe = Join-Path $projectRoot ".venv_yolo\Scripts\python.exe"
$statsPy = Join-Path $projectRoot "spic_submission\revision_stats_from_csv.py"

$baselineCsv = Join-Path $repoRoot "results\benchmarks\uieb\models\euvp_cyclegan_continue250_from200_s0\epoch_250\per_image.csv"
$candidateCsv = Join-Path $repoRoot "results\benchmarks\uieb\models\euvp_mpcgan_continue250_from200_s0\epoch_250\per_image.csv"
$outputJson = Join-Path $projectRoot "spic_submission\revision_stats_uieb_250_vs_250.json"

Write-Host "Waiting for candidate CSV: $candidateCsv"
while (-not (Test-Path $candidateCsv)) {
    Start-Sleep -Seconds 30
}

Write-Host "Candidate CSV detected. Computing paired statistics..."
& $pythonExe $statsPy `
    --baseline $baselineCsv `
    --candidate $candidateCsv `
    --label-baseline "CycleGAN (+50-epoch continuation)" `
    --label-candidate "MP-CycleGAN (+50-epoch refinement)" `
    --output-json $outputJson

Write-Host "Paired statistics written to: $outputJson"
