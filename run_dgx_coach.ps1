$sshExecutable = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$sshHost = "dgx-spark"
$localPort = 18765
$remotePort = 18765
$remoteScript = "/home/boni/projects/MineStudio/run/run_dgx_coach_server.sh"
$remoteStopScript = "/home/boni/projects/MineStudio/run/stop_dgx_coach_server.sh"
$remoteWaitScript = "/home/boni/projects/MineStudio/run/wait_dgx_coach_server.sh"
$viewerPython = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$viewerScript = "D:\HugeProjects\MineStudio\run\gemini_coach_viewer.py"
$transportTestScript = "D:\HugeProjects\MineStudio\run\test_dgx_coach_transport.py"
$headlessTest = $env:MINESTUDIO_COACH_HEADLESS_TEST -eq "1"
$clientScript = if ($headlessTest) { $transportTestScript } else { $viewerScript }
$logDirectory = "D:\HugeProjects\MineStudio\output"
$standardOutputLog = "$logDirectory\dgx-coach-server.stdout.log"
$standardErrorLog = "$logDirectory\dgx-coach-server.stderr.log"
$viewerOutputLog = "$logDirectory\dgx-coach-viewer.stdout.log"
$viewerErrorLog = "$logDirectory\dgx-coach-viewer.stderr.log"
$sshControlArguments = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ConnectionAttempts=1",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=3"
)
$sshArguments = @(
    $sshControlArguments
    "-o", "ExitOnForwardFailure=yes",
    "-L", "${localPort}:127.0.0.1:${remotePort}",
    $sshHost,
    "bash", $remoteScript
)

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Write-Host "[1/4] Removing any abandoned MineStudio session on the DGX..."
$cleanupOutput = & $sshExecutable @sshControlArguments $sshHost "bash" $remoteStopScript 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Could not clean up the previous DGX coach server.`n$cleanupOutput"
}

$existingListeners = @(Get-NetTCPConnection -LocalPort $localPort -State Listen -ErrorAction SilentlyContinue)
foreach ($existingListener in $existingListeners) {
    $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($existingListener.OwningProcess)"
    $forwardSpec = "${localPort}:127.0.0.1:${remotePort}"
    $isMineStudioTunnel = $existingProcess.Name -eq "ssh.exe" -and $existingProcess.CommandLine -like "*$forwardSpec*" -and $existingProcess.CommandLine -like "*$remoteScript*"
    if (-not $isMineStudioTunnel) {
        throw "Local port $localPort is used by PID $($existingListener.OwningProcess), which is not a MineStudio tunnel."
    }
    Stop-Process -Id $existingListener.OwningProcess
    Wait-Process -Id $existingListener.OwningProcess -Timeout 10 -ErrorAction SilentlyContinue
}

Write-Host "[2/4] Opening the encrypted SSH tunnel..."
$serverProcess = Start-Process -FilePath $sshExecutable -ArgumentList $sshArguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $standardOutputLog -RedirectStandardError $standardErrorLog

try {
    Write-Host "[3/4] Waiting up to 120 seconds for the DGX server to bind its port..."
    $readyOutput = & $sshExecutable @sshControlArguments $sshHost "bash" $remoteWaitScript 2>&1
    $serverProcess.Refresh()
    if ($LASTEXITCODE -ne 0 -or $serverProcess.HasExited) {
        $serverError = Get-Content -Raw $standardErrorLog -ErrorAction SilentlyContinue
        throw "The DGX coach server failed its readiness check.`n$readyOutput`n$serverError"
    }
    Write-Host "[4/4] The DGX server is ready. Opening the Minecraft viewer..."
    if ($headlessTest) {
        & $viewerPython $clientScript
    } else {
        $viewerProcess = Start-Process -FilePath $viewerPython -ArgumentList $clientScript -PassThru -RedirectStandardOutput $viewerOutputLog -RedirectStandardError $viewerErrorLog
        Write-Host "Viewer PID $($viewerProcess.Id) started. Close the viewer window to end the session."
        $viewerProcess.WaitForExit()
        if ($viewerProcess.ExitCode -ne 0) {
            $viewerError = Get-Content -Raw $viewerErrorLog -ErrorAction SilentlyContinue
            throw "The Minecraft viewer exited with code $($viewerProcess.ExitCode).`n$viewerError"
        }
    }
} finally {
    if (-not $serverProcess.HasExited) {
        $serverProcess.WaitForExit(15000) | Out-Null
    }
    if (-not $serverProcess.HasExited) {
        & $sshExecutable @sshControlArguments $sshHost "bash" $remoteStopScript | Out-Null
        $serverProcess.WaitForExit(10000) | Out-Null
    }
    if (-not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id
        $serverProcess.WaitForExit(10000) | Out-Null
    }
    & $sshExecutable @sshControlArguments $sshHost "bash" $remoteStopScript | Out-Null
    Write-Host "MineStudio session closed."
}
