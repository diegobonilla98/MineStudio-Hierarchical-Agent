$distribution = "Ubuntu-24.04"
$serverScript = "/mnt/d/HugeProjects/MineStudio/run_wsl_coach_server.sh"
$viewerPython = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$viewerScript = "D:\HugeProjects\MineStudio\run\gemini_coach_viewer.py"
$logDirectory = "D:\HugeProjects\MineStudio\output"
$standardOutputLog = "$logDirectory\coach-server.stdout.log"
$standardErrorLog = "$logDirectory\coach-server.stderr.log"
$untranslatablePath = "E:\Tools\COLMAP\4.0.4-cuda\bin"
$originalPath = $env:PATH
$env:PATH = (($originalPath -split ";") | Where-Object { $_ -ne $untranslatablePath }) -join ";"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

$serverProcess = Start-Process -FilePath "$env:WINDIR\System32\wsl.exe" -ArgumentList @("-d", $distribution, "--", "bash", $serverScript) -PassThru -WindowStyle Hidden -RedirectStandardOutput $standardOutputLog -RedirectStandardError $standardErrorLog

try {
    & $viewerPython $viewerScript
} finally {
    if (-not $serverProcess.HasExited) {
        $serverProcess.WaitForExit(120000) | Out-Null
    }
    if (-not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force
    }
    $env:PATH = $originalPath
}
