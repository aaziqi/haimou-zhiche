param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [string]$ReferenceDir = "",
    [string]$CheckpointName = "euvp_cyclegan_full",
    [string]$Epoch = "latest",
    [string]$Direction = "AtoB",
    [string]$GpuIds = "-1",
    [string]$OutputDir = "",
    [string]$ReportTitle = "Underwater Image Enhancement Demo",
    [string]$TrackName = "AI Application",
    [int]$NumTest = 0,
    [switch]$Overwrite
)

function Resolve-PythonExe {
    $candidates = @()
    try {
        $pythonCommand = Get-Command python.exe -ErrorAction Stop
        if ($pythonCommand.Source -notmatch "WindowsApps") {
            $candidates += $pythonCommand.Source
        }
    } catch {
    }
    $candidates += @(
        "C:\Users\86157\anaconda3\envs\pytorch\python.exe",
        "C:\Users\86157\anaconda3\python.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }
    throw "Python interpreter not found. Activate the environment or update the candidate paths in this script."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Resolve-PythonExe
$demoScript = Join-Path $PSScriptRoot "underwater_competition_demo.py"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "competition_demo_outputs\national_award_showcase"
}

$arguments = @(
    $demoScript,
    "--input", $InputPath,
    "--checkpoint_name", $CheckpointName,
    "--epoch", $Epoch,
    "--direction", $Direction,
    "--gpu_ids", $GpuIds,
    "--output_dir", $OutputDir,
    "--report_title", $ReportTitle,
    "--track_name", $TrackName
)

if (-not [string]::IsNullOrWhiteSpace($ReferenceDir)) {
    $arguments += @("--reference_dir", $ReferenceDir)
}

if ($NumTest -gt 0) {
    $arguments += @("--num_test", $NumTest)
}

if ($Overwrite) {
    $arguments += "--overwrite"
}

Push-Location $repoRoot
try {
    & $pythonExe @arguments
} finally {
    Pop-Location
}
