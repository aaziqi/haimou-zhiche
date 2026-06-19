$ErrorActionPreference = "Stop"

$projectRoot = "D:\VScode\Graduation project"
$repoRoot = Join-Path $projectRoot "pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master"
$pythonExe = Join-Path $projectRoot ".venv_yolo\Scripts\python.exe"
$trainPy = Join-Path $repoRoot "train.py"
$dataRoot = Join-Path $projectRoot "EUVP_Unpaired"
$logDir = Join-Path $projectRoot "spic_submission\training_logs"

$baselineDst = Join-Path $repoRoot "checkpoints\euvp_cyclegan_continue250_from200_s0"
$mpSrc = Join-Path $repoRoot "checkpoints\euvp_mpcgan_stage2_s0"
$mpDstName = "euvp_mpcgan_continue250_from200_s0"
$mpDst = Join-Path $repoRoot ("checkpoints\" + $mpDstName)
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

$targetCheckpoint = Join-Path $baselineDst "250_net_G_A.pth"
Write-Host "Waiting for baseline checkpoint: $targetCheckpoint"
while (-not (Test-Path $targetCheckpoint)) {
    Start-Sleep -Seconds 30
}

Write-Host "Baseline reached epoch 250. Preparing MP continuation..."
Copy-CheckpointSet -SourceDir $mpSrc -DestDir $mpDst -Patterns @(
    "200_net_*.pth",
    "201_net_*.pth",
    "202_net_*.pth",
    "203_net_*.pth",
    "latest_net_*.pth",
    "train_opt.txt",
    "test_opt.txt"
)

Push-Location $repoRoot
try {
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
}
finally {
    Pop-Location
}
