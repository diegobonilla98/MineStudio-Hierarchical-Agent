$ErrorActionPreference = "Stop"

$state = ssh dgx-spark 'root=/home/boni/projects/MineStudio/output/iron_pickaxe_viability/latest; if test -f "$root/run_state.json"; then cat "$root/run_state.json"; else echo "{}"; fi'
$parsed = $state | ConvertFrom-Json
if (-not $parsed.status) {
    Write-Output "state=not_started"
    exit 0
}

Write-Output "state=$($parsed.status) phase=$($parsed.phase) detail=$($parsed.detail) runner_pid=$($parsed.pid)"
foreach ($arm in @("full", "no_specialist", "no_system2")) {
    $line = ssh dgx-spark "root=/home/boni/projects/MineStudio/output/iron_pickaxe_viability/latest; if test -f `$root/$arm/summary.json; then jq -r '[.episodes_valid,.episodes_target,.successes,.success_rate] | @tsv' `$root/$arm/summary.json; else echo '0`t24`t0`t0'; fi"
    $values = $line -split "`t"
    $percent = [math]::Round(([double]$values[3]) * 100, 1)
    Write-Output "$arm=$($values[0])/$($values[1]) iron_pickaxe=$($values[2]) ($percent%)"
}

$recent = ssh dgx-spark 'root=/home/boni/projects/MineStudio/output/iron_pickaxe_viability/latest; for arm in full no_specialist no_system2; do echo "[$arm]"; tail -n 2 "$root/logs/$arm.log" 2>/dev/null || true; done'
if ($recent) {
    Write-Output "recent logs:"
    Write-Output $recent
}
