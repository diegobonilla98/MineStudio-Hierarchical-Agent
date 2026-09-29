$ErrorActionPreference = "Stop"
$server = "dgx-spark"
$projectDirectory = "/home/boni/projects/MineStudio"
$outputDirectory = "$projectDirectory/output/stone_acquisition/stage0_baseline_v1"
$statePath = "$outputDirectory/runner_state.json"
$episodesPath = "$outputDirectory/episodes.jsonl"
$summaryPath = "$outputDirectory/summary.json"
$runnerLog = "$outputDirectory/runner.log"
$matrixEpisodesPath = "$projectDirectory/output/steve_skill_matrix/controlled_15_skill_100/episodes.jsonl"
$stateJson = ((& ssh $server "cat $statePath 2>/dev/null || true") -join "`n").Trim()
if ($stateJson) {
    $state = $stateJson | ConvertFrom-Json
} else {
    $state = [pscustomobject]@{ state = "not_queued"; runner_pid = $null }
}
$counts = (& ssh $server "baseline=0; matrix=0; test -f $episodesPath && baseline=`$(wc -l < $episodesPath); test -f $matrixEpisodesPath && matrix=`$(wc -l < $matrixEpisodesPath); printf '%s %s' `"`$baseline`" `"`$matrix`"").Trim().Split(' ')
$baselineEpisodes = [int]$counts[0]
$matrixEpisodes = [int]$counts[1]
$baselinePercent = [math]::Round(100.0 * $baselineEpisodes / 1000, 2)
$matrixPercent = [math]::Round(100.0 * $matrixEpisodes / 1500, 2)
$processes = (& ssh $server "pgrep -af '[b]enchmark_stone_acquisition.py|[b]enchmark_steve_skill_matrix.py|[r]un_stone_baseline_after_matrix.sh' || true") -join "`n"
Write-Host "state=$($state.state) runner_pid=$($state.runner_pid)"
Write-Host "skill_matrix=$matrixEpisodes/1500 ($matrixPercent%)"
Write-Host "stone_baseline=$baselineEpisodes/1000 ($baselinePercent%)"
Write-Host "summary=$summaryPath"
if ($processes) {
    Write-Host "processes:"
    Write-Host $processes
}
$recent = (& ssh $server "tail -n 8 $runnerLog 2>/dev/null || true") -join "`n"
if ($recent) {
    Write-Host "recent baseline log:"
    Write-Host $recent
}
