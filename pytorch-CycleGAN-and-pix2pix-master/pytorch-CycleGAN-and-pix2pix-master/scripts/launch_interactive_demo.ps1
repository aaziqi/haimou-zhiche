param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8877,
    [string]$GpuIds = "-1"
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
$serverScript = Join-Path $PSScriptRoot "interactive_demo_server.py"

Push-Location $repoRoot
try {
    & $pythonExe $serverScript --host $BindHost --port $Port --gpu_ids $GpuIds
} finally {
    Pop-Location
}
