$RemoteCommand = @'
set -e
project_directory=/home/boni/projects/MineStudio
output_directory=$project_directory/output/stone_recovery_bc/latest_run
dataset_summary=$project_directory/output/stone_recovery_bc/dataset_v1/summary.json

if ! test -e "$output_directory"; then
    echo state=not_started
    exit 0
elif test -f "$output_directory/sweep.complete"; then
    echo state=completed
elif pgrep -f '[r]un_stone_recovery_bc_sweep.sh' >/dev/null; then
    echo state=running
else
    echo state=stopped
fi

echo "run_directory=$(readlink -f "$output_directory")"

if test -f "$dataset_summary"; then
    dataset_windows=$(jq -r '.windows' "$dataset_summary")
    dataset_steps=$(jq -r '.window_steps' "$dataset_summary")
    echo "dataset_windows=$dataset_windows dataset_steps=$dataset_steps"
fi

for variant in action_head action_head_recurrent upper_lora; do
    summary=$output_directory/$variant/summary.json
    variant_state=$output_directory/$variant/run_state.json
    metrics=$output_directory/$variant/metrics.jsonl
    if test -f "$summary"; then
        completed_steps=$(jq -r '.completed_steps' "$summary")
        best_step=$(jq -r '.best_step' "$summary")
        validation_loss=$(jq -r '.final_validation.loss' "$summary")
        echo "variant=$variant steps=$completed_steps best_step=$best_step val_loss=$validation_loss"
    elif test -f "$variant_state"; then
        variant_status=$(jq -r '.status' "$variant_state")
        current_step=$(jq -r '.step // ((.start_step // 1) - 1)' "$variant_state")
        if test -s "$metrics"; then
            current_step=$(tail -n 1 "$metrics" | jq -r '.step')
        fi
        max_steps=$(jq -r '.max_steps // 400' "$variant_state")
        echo "variant=$variant status=$variant_status step=$current_step/$max_steps"
    else
        echo "variant=$variant pending"
    fi
done

if test -f "$output_directory/selection.json"; then
    selected=$(jq -r '.selected.variant' "$output_directory/selection.json")
    selected_loss=$(jq -r '.selected.validation_loss' "$output_directory/selection.json")
    echo "selected=$selected selected_val_loss=$selected_loss"
fi

evaluation_summary=$(find "$project_directory/output/stone_acquisition" -maxdepth 2 -path '*/stage1_bc_*_v1/summary.json' -print -quit)
if test -n "$evaluation_summary"; then
    evaluation_complete=$(jq -r '.episodes_complete' "$evaluation_summary")
    evaluation_target=$(jq -r '.episodes_target' "$evaluation_summary")
    evaluation_success=$(jq -r '.overall.success_rate' "$evaluation_summary")
    echo "evaluation=$evaluation_complete/$evaluation_target success_rate=$evaluation_success"
fi

if test -f "$output_directory/sweep.log"; then
    echo recent_log:
    tail -n 12 "$output_directory/sweep.log"
fi
'@

ssh dgx-spark $RemoteCommand
