$ErrorActionPreference = "Stop"

$projectRoot = "D:\VScode\Graduation project"
$repoRoot = Join-Path $projectRoot "pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master"
$pythonExe = Join-Path $projectRoot ".venv_yolo\Scripts\python.exe"
$trainPy = Join-Path $repoRoot "train.py"
$dataRoot = Join-Path $projectRoot "EUVP_Unpaired"
$logDir = Join-Path $projectRoot "spic_submission\training_logs"

$baselineSrc = Join-Path $repoRoot "checkpoints\euvp_cyclegan_full"
$baselineDstName = "euvp_cyclegan_continue250_from200_s0"
$baselineDst = Join-Path $repoRoot ("checkpoints\" + $baselineDstName)

$mpSrc = Join-Path $repoRoot "checkpoints\euvp_mpcgan_stage2_s0"
$mpDstName = "euvp_mpcgan_continue250_from200_s0"
$mpDst = Join-Path $repoRoot ("checkpoints\" + $mpDstName)

$baselineConsoleLog = Join-Path $logDir "baseline_to250_console.log"
$mpConsoleLog = Join-Path $logDir "mp_to250_console.log"

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Copy-CheckpointSet {
    param(
        [string]$SourceDir,
        [string]$DestDir,
        [string[]]$Patterns
    )
    Ensure-Dir -Path $DestDir
    foreach ($pattern in $Patterns) {
        Get-ChildItem -Path $SourceDir -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName -Destination (Join-Path $DestDir $_.Name) -Force
        }
    }
}

Ensure-Dir -Path $logDir

if (-not (Test-Path $pythonExe)) {
    throw "Python interpreter not found: $pythonExe"
}

if (-not (Test-Path $trainPy)) {
    throw "train.py not found: $trainPy"
}

if (-not (Test-Path $dataRoot)) {
    throw "Dataset root not found: $dataRoot"
}

Push-Location $repoRoot
try {
    Write-Host "Preparing continuation checkpoint folders..."

    Copy-CheckpointSet -SourceDir $baselineSrc -DestDir $baselineDst -Patterns @(
        "200_net_*.pth",
        "latest_net_*.pth",
        "train_opt.txt",
        "test_opt.txt"
    )

    Copy-CheckpointSet -SourceDir $mpSrc -DestDir $mpDst -Patterns @(
        "200_net_*.pth",
        "201_net_*.pth",
        "202_net_*.pth",
        "203_net_*.pth",
        "latest_net_*.pth",
        "train_opt.txt",
        "test_opt.txt"
    )

    Write-Host "Starting baseline continuation to epoch 250..."
    & $pythonExe $trainPy `
        --dataroot $dataRoot `
        --name $baselineDstName `
        --model cycle_gan `
        --continue_train `
        --epoch 200 `
        --epoch_count 201 `
        --n_epochs 250 `
        --n_epochs_decay 0 `
        --lr 0.0001 `
        --batch_size 1 `
        --preprocess resize_and_crop `
        --load_size 286 `
        --crop_size 256 `
        --netG resnet_9blocks `
        --netD basic `
        --no_dropout `
        --max_dataset_size 500 `
        --seed 0 `
        --print_freq 100 `
        --save_epoch_freq 1 `
        --display_freq 1000000000 `
        --update_html_freq 1000000000 `
        --no_html 2>&1 | Tee-Object -FilePath $baselineConsoleLog -Append

    Write-Host "Baseline continuation finished."
    Write-Host "Starting MP continuation to epoch 250..."
    & $pythonExe $trainPy `
        --dataroot $dataRoot `
        --name $mpDstName `
        --model cycle_gan `
        --continue_train `
        --epoch 203 `
        --epoch_count 204 `
        --n_epochs 250 `
        --n_epochs_decay 0 `
        --lr 0.0001 `
        --batch_size 1 `
        --preprocess resize_and_crop `
        --load_size 286 `
        --crop_size 256 `
        --netG resnet_9blocks `
        --netD basic `
        --no_dropout `
        --max_dataset_size 500 `
        --seed 0 `
        --print_freq 100 `
        --save_epoch_freq 1 `
        --display_freq 1000000000 `
        --update_html_freq 1000000000 `
        --no_html `
        --lambda_gray 0.1 `
        --lambda_struct 2.0 `
        --lambda_perceptual 0.05 `
        --lambda_color 0.0 `
        --perceptual_layer 16 `
        --perceptual_weights imagenet 2>&1 | Tee-Object -FilePath $mpConsoleLog -Append

    Write-Host "MP continuation finished."
    Write-Host "Logs:"
    Write-Host "  Baseline console: $baselineConsoleLog"
    Write-Host "  MP console:       $mpConsoleLog"
    Write-Host "  Baseline loss:    $(Join-Path $baselineDst 'loss_log.txt')"
    Write-Host "  MP loss:          $(Join-Path $mpDst 'loss_log.txt')"
}
finally {
    Pop-Location
}
