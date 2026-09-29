$remoteRoot = "/home/boni/projects/MineStudio/output/stone_acquisition/latest_system0_isolation"
$stateText = ssh dgx-spark "cat '$remoteRoot/run_state.json' 2>/dev/null || true"
if (-not $stateText) {
    Write-Output "state=not_started"
    exit 0
}
$state = $stateText | ConvertFrom-Json
Write-Output "output_directory=$($state.output_directory)"
Write-Output "state=$($state.status) phase=$($state.phase) configuration=$($state.configuration) split=$($state.split)"
$experiments = @(
    @{ Name = "old_router_hardened_pickup/normal"; Directory = "system0_isolation_normal_old_router_hardened_pickup_v1"; Target = 40 },
    @{ Name = "old_router_hardened_pickup/hazard"; Directory = "system0_isolation_hazard_old_router_hardened_pickup_v1"; Target = 60 },
    @{ Name = "new_router_old_pickup/normal"; Directory = "system0_isolation_normal_new_router_old_pickup_v1"; Target = 40 },
    @{ Name = "new_router_old_pickup/hazard"; Directory = "system0_isolation_hazard_new_router_old_pickup_v1"; Target = 60 }
)
foreach ($experiment in $experiments) {
    $summaryText = ssh dgx-spark "cat '/home/boni/projects/MineStudio/output/stone_acquisition/$($experiment.Directory)/summary.json' 2>/dev/null || true"
    if ($summaryText) {
        $summary = $summaryText | ConvertFrom-Json
        Write-Output "$($experiment.Name)=$($summary.episodes_complete)/$($experiment.Target) success_rate=$($summary.overall.success_rate)"
    } else {
        $countText = ssh dgx-spark "test -f '/home/boni/projects/MineStudio/output/stone_acquisition/$($experiment.Directory)/episodes.jsonl' && wc -l < '/home/boni/projects/MineStudio/output/stone_acquisition/$($experiment.Directory)/episodes.jsonl' || echo 0"
        Write-Output "$($experiment.Name)=$($countText.Trim())/$($experiment.Target)"
    }
}
$reportText = ssh dgx-spark "cat '$remoteRoot/system0_isolation_report.json' 2>/dev/null || true"
if ($reportText) {
    $report = $reportText | ConvertFrom-Json
    foreach ($name in @("old_router_old_pickup", "old_router_hardened_pickup", "new_router_old_pickup", "new_router_hardened_pickup")) {
        $metric = $report.combined.metrics.$name
        Write-Output "$name combined_success=$($metric.success_rate) water_success=$($metric.success_given_entered_water) reacquisition=$($metric.stone_reacquisition_after_water_rate) transitions=$($metric.mean_router_transitions)"
    }
}
$criticText = ssh dgx-spark "cat '$remoteRoot/gemini_trajectory_critic_summary.json' 2>/dev/null || true"
if ($criticText) {
    $critic = $criticText | ConvertFrom-Json
    Write-Output "critic_status=$($critic.status) critic_saved=$($critic.critiques)/$($critic.target)"
}
$logName = if ($state.phase -eq "evaluation") { "$($state.configuration)_$($state.split).log" } elseif ($state.phase -eq "gemini_critic") { "gemini_critic.log" } else { $null }
if ($logName) {
    Write-Output "recent_log:"
    ssh dgx-spark "tail -n 12 '$remoteRoot/$logName' 2>/dev/null || true"
}
