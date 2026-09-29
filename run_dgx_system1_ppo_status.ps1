$ErrorActionPreference = "Stop"

$pythonScript = @'
import json
from pathlib import Path


root = Path("/home/boni/projects/MineStudio/output/system1_recovery_ppo/latest")


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


metrics = []
metrics_path = root / "metrics.jsonl"
if metrics_path.is_file():
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
print(json.dumps({
    "pipeline": read_json(root / "pipeline_state.json"),
    "training": read_json(root / "run_state.json"),
    "latest_metric": metrics[-1] if metrics else {},
    "final": read_json(root / "final_report.json"),
}))
'@
$encodedScript = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pythonScript))
$remoteCommand = "/home/boni/ai/envs/dgx-dl/bin/python -c `"import base64;exec(base64.b64decode('$encodedScript'))`""
$payloadText = ssh dgx-spark $remoteCommand
$payload = $payloadText | ConvertFrom-Json
$state = $payload.pipeline
if (-not $state.status) {
    Write-Output "state=not_started"
    exit 0
}

Write-Output "state=$($state.status) phase=$($state.phase) detail=$($state.detail) runner_pid=$($state.pid)"
$training = $payload.training
if ($training.status) {
    $eta = if ($training.estimated_remaining_seconds) { [math]::Round([double]$training.estimated_remaining_seconds / 60, 1) } else { 0 }
    Write-Output "training=$($training.status) iteration=$($training.iteration)/$($training.max_iterations) best=$($training.best_iteration) eta_minutes=$eta"
    if ($training.latest_development) {
        $weak = [math]::Round([double]$training.latest_development.weak_mean * 100, 1)
        $normal = [math]::Round([double]$training.latest_development.normal_stone.success_rate * 100, 1)
        Write-Output "development_weak=$weak% normal_stone=$normal%"
    }
}

$metric = $payload.latest_metric
if ($metric.split) {
    Write-Output "latest_metric=iteration_$($metric.iteration)_$($metric.split)"
}

$final = $payload.final
if ($final.promotion) {
    $success = [math]::Round([double]$final.stone.combined.candidate_success_rate * 100, 1)
    $difference = [math]::Round([double]$final.stone.combined.difference * 100, 1)
    Write-Output "final_promoted=$($final.promotion.promoted) stone_success=$success% difference=$difference`pp"
}
