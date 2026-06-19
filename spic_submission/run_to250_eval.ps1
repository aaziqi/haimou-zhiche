$ErrorActionPreference = "Stop"

$projectRoot = "D:\VScode\Graduation project"
$repoRoot = Join-Path $projectRoot "pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master"
$pythonExe = Join-Path $projectRoot ".venv_yolo\Scripts\python.exe"
$testPy = Join-Path $repoRoot "test.py"
$evalPy = Join-Path $repoRoot "scripts\evaluate_euvp_psnr_ssim.py"
$statsPy = Join-Path $projectRoot "spic_submission\revision_stats_from_csv.py"

$euvpInp = Join-Path $projectRoot "EUVP Dataset\test_samples\Inp"
$euvpRef = Join-Path $projectRoot "EUVP Dataset\test_samples\GTr"
$uiebInp = Join-Path $projectRoot "UIEB\raw-890-s\raw-890\raw-890"
$uiebRef = Join-Path $projectRoot "UIEB\reference-890\reference-890"

$baselineName = "euvp_cyclegan_continue250_from200_s0"
$mpName = "euvp_mpcgan_continue250_from200_s0"
$epoch = "250"

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Run-Inference {
    param(
        [string]$DataRoot,
        [string]$ModelName,
        [string]$ResultsDir,
        [int]$NumTest
    )
    & $pythonExe $testPy `
        --dataroot $DataRoot `
        --name $ModelName `
        --model test `
        --dataset_mode single `
        --netG resnet_9blocks `
        --norm instance `
        --results_dir $ResultsDir `
        --model_suffix "_A" `
        --epoch $epoch `
        --num_test $NumTest
}

function Run-Eval {
    param(
        [string]$InpDir,
        [string]$RefDir,
        [string]$PredDir,
        [string]$CsvPath,
        [string]$JsonPath
    )
    Ensure-Dir -Path (Split-Path -Parent $CsvPath)
    Ensure-Dir -Path (Split-Path -Parent $JsonPath)
    & $pythonExe $evalPy `
        --inp_dir $InpDir `
        --ref_dir $RefDir `
        --pred_dir $PredDir `
        --save_csv $CsvPath `
        --save_json $JsonPath `
        --bootstrap_iters 2000 `
        --seed 123
}

Push-Location $repoRoot
try {
    Write-Host "Running EUVP inference..."
    Run-Inference -DataRoot $euvpInp -ModelName $baselineName -ResultsDir ".\results\euvp" -NumTest 10000
    Run-Inference -DataRoot $euvpInp -ModelName $mpName -ResultsDir ".\results\euvp" -NumTest 10000

    Write-Host "Running UIEB inference..."
    Run-Inference -DataRoot $uiebInp -ModelName $baselineName -ResultsDir ".\results\uieb" -NumTest 10000
    Run-Inference -DataRoot $uiebInp -ModelName $mpName -ResultsDir ".\results\uieb" -NumTest 10000

    Write-Host "Evaluating EUVP..."
    Run-Eval `
        -InpDir $euvpInp `
        -RefDir $euvpRef `
        -PredDir (Join-Path $repoRoot "results\euvp\$baselineName\test_250\images") `
        -CsvPath (Join-Path $repoRoot "results\benchmarks\euvp\models\$baselineName\epoch_250\per_image.csv") `
        -JsonPath (Join-Path $repoRoot "results\benchmarks\euvp\models\$baselineName\epoch_250\summary.json")
    Run-Eval `
        -InpDir $euvpInp `
        -RefDir $euvpRef `
        -PredDir (Join-Path $repoRoot "results\euvp\$mpName\test_250\images") `
        -CsvPath (Join-Path $repoRoot "results\benchmarks\euvp\models\$mpName\epoch_250\per_image.csv") `
        -JsonPath (Join-Path $repoRoot "results\benchmarks\euvp\models\$mpName\epoch_250\summary.json")

    Write-Host "Evaluating UIEB..."
    Run-Eval `
        -InpDir $uiebInp `
        -RefDir $uiebRef `
        -PredDir (Join-Path $repoRoot "results\uieb\$baselineName\test_250\images") `
        -CsvPath (Join-Path $repoRoot "results\benchmarks\uieb\models\$baselineName\epoch_250\per_image.csv") `
        -JsonPath (Join-Path $repoRoot "results\benchmarks\uieb\models\$baselineName\epoch_250\summary.json")
    Run-Eval `
        -InpDir $uiebInp `
        -RefDir $uiebRef `
        -PredDir (Join-Path $repoRoot "results\uieb\$mpName\test_250\images") `
        -CsvPath (Join-Path $repoRoot "results\benchmarks\uieb\models\$mpName\epoch_250\per_image.csv") `
        -JsonPath (Join-Path $repoRoot "results\benchmarks\uieb\models\$mpName\epoch_250\summary.json")

    Write-Host "Computing paired statistics..."
    & $pythonExe $statsPy `
        --baseline (Join-Path $repoRoot "results\benchmarks\euvp\models\$baselineName\epoch_250\per_image.csv") `
        --candidate (Join-Path $repoRoot "results\benchmarks\euvp\models\$mpName\epoch_250\per_image.csv") `
        --label-baseline "CycleGAN (+50-epoch continuation)" `
        --label-candidate "MP-CycleGAN (+50-epoch refinement)" `
        --output-json (Join-Path $projectRoot "spic_submission\revision_stats_euvp_250_vs_250.json")
    & $pythonExe $statsPy `
        --baseline (Join-Path $repoRoot "results\benchmarks\uieb\models\$baselineName\epoch_250\per_image.csv") `
        --candidate (Join-Path $repoRoot "results\benchmarks\uieb\models\$mpName\epoch_250\per_image.csv") `
        --label-baseline "CycleGAN (+50-epoch continuation)" `
        --label-candidate "MP-CycleGAN (+50-epoch refinement)" `
        --output-json (Join-Path $projectRoot "spic_submission\revision_stats_uieb_250_vs_250.json")
}
finally {
    Pop-Location
}
