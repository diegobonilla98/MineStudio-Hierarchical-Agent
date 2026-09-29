$ErrorActionPreference = "Stop"

$state = ssh dgx-spark 'root=/home/boni/projects/MineStudio/output/system1_recovery_micro/latest; if test -f "$root/run_state.json"; then cat "$root/run_state.json"; else echo "{}"; fi'
$parsed = $state | ConvertFrom-Json
if (-not $parsed.status) {
    Write-Output "state=not_started"
    exit 0
}

Write-Output "state=$($parsed.status) phase=$($parsed.phase) task=$($parsed.task) runner_pid=$($parsed.pid)"
$tasks = @("exit_water", "climb_shore", "reacquire_stone", "escape_hole", "avoid_digging_trap", "recover_camera", "avoid_water")
foreach ($task in $tasks) {
    $line = ssh dgx-spark "root=/home/boni/projects/MineStudio/output/system1_recovery_micro/latest; if test -f `$root/$task/summary.json; then jq -r '[.episodes_complete,.episodes_target,.overall.successes,.overall.success_rate] | @tsv' `$root/$task/summary.json; else echo '0`t50`t0`t0'; fi"
    $values = $line -split "`t"
    $percent = [math]::Round(([double]$values[3]) * 100, 1)
    Write-Output "$task=$($values[0])/$($values[1]) success=$($values[2]) ($percent%)"
}

$critic = ssh dgx-spark 'root=/home/boni/projects/MineStudio/output/system1_recovery_micro/latest; if test -f "$root/gemini_critic_summary.json"; then jq -r '"'"'[.status,.critiques,.target,.remaining] | @tsv'"'"' "$root/gemini_critic_summary.json"; fi'
if ($critic) {
    $values = $critic -split "`t"
    Write-Output "gemini=$($values[0]) critiques=$($values[1])/$($values[2]) remaining=$($values[3])"
}
