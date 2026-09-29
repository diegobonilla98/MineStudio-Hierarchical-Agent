$remote = @'
cd /home/boni/projects/MineStudio
directory=$(readlink -f output/stone_recovery_bc/latest_architecture_sweep 2>/dev/null || true)
if test -z "$directory" || ! test -d "$directory"; then
    echo state=not_started
    exit 0
fi
echo output_directory=$directory
if test -f "$directory/run_state.json"; then
    jq -r '"state=" + .status + " variant=" + .variant + " runner_pid=" + (.pid|tostring)' "$directory/run_state.json"
fi
for variant in upper_lora upper_lora_rank32 recurrent_upper_lora upper_blocks_full policy_no_visual_full; do
    state_file="$directory/$variant/run_state.json"
    if test -f "$state_file"; then
        status=$(jq -r '.status' "$state_file")
        step=$(jq -r '.step // empty' "$state_file")
        if test -z "$step" && test -f "$directory/$variant.log"; then
            step=$(grep '"status": "training"' "$directory/$variant.log" | tail -n 1 | jq -r '.step' 2>/dev/null || true)
        fi
        step=${step:-0}
        best_step=$(jq -r '.best_step // 0' "$state_file")
        best_loss=$(jq -r '.best_validation_loss // "pending"' "$state_file")
        echo "$variant=$status step=$step/6400 best_step=$best_step best_loss=$best_loss"
    else
        echo "$variant=pending"
    fi
done
echo recent_log:
active=$(jq -r '.variant // empty' "$directory/run_state.json" 2>/dev/null || true)
if test -n "$active" && test -f "$directory/$active.log"; then
    tail -n 8 "$directory/$active.log"
fi
'@
ssh dgx-spark $remote
