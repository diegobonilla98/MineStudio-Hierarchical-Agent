$ErrorActionPreference = "Stop"
$server = "dgx-spark"
$projectDirectory = "/home/boni/projects/MineStudio"
$runnerScript = "$projectDirectory/run/run_stone_baseline_after_matrix.sh"
$outputDirectory = "$projectDirectory/output/stone_acquisition/stage0_baseline_v1"
$launcherLog = "$outputDirectory/launcher.log"
$statePath = "$outputDirectory/runner_state.json"
$launchCommand = "cd $projectDirectory; mkdir -p $outputDirectory; chmod +x $runnerScript; setsid -f bash $runnerScript >> $launcherLog 2>&1 < /dev/null"
& ssh $server $launchCommand
if ($LASTEXITCODE -ne 0) {
    throw "DGX queue launch command failed with exit code $LASTEXITCODE."
}
Start-Sleep -Seconds 3
$stateJson = (& ssh $server "if test -f $statePath; then cat $statePath; else echo '{}'; fi") -join "`n"
$state = $stateJson | ConvertFrom-Json
$stateRunnerCheck = "exited"
if ($state.runner_pid -match '^\d+$') {
    $stateRunnerCheck = (& ssh $server "if kill -0 $($state.runner_pid) 2>/dev/null; then echo alive; else echo exited; fi").Trim()
}
if ($stateRunnerCheck -eq "alive") {
    Write-Host "Stone baseline queue is active with runner PID $($state.runner_pid)."
} elseif ($state.state) {
    Write-Host "Stone baseline runner exited after reporting state '$($state.state)'."
} else {
    $launcherTail = (& ssh $server "tail -n 20 $launcherLog 2>/dev/null || true") -join "`n"
    throw "Stone baseline queue failed before writing state.`n$launcherTail"
}
if ($state.state) {
    Write-Host "state=$($state.state) matrix_pid=$($state.matrix_pid)"
}
Write-Host "Status: & `"D:\HugeProjects\MineStudio\run_dgx_stone_baseline_status.ps1`""
