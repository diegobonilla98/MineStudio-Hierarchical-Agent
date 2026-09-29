#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
RUN_ID=${MINESTUDIO_BC_RUN_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
OUTPUT_DIRECTORY=${MINESTUDIO_BC_OUTPUT_DIRECTORY:-$PROJECT_DIRECTORY/output/stone_recovery_bc/architecture_sweeps/$RUN_ID}
RUN_STATE=$OUTPUT_DIRECTORY/run_state.json
TASK_BANK=$PROJECT_DIRECTORY/output/stone_recovery_bc/dataset_v1/gemini_task_bank.json
VARIANTS=(upper_lora upper_lora_rank32 recurrent_upper_lora upper_blocks_full policy_no_visual_full)

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
mkdir -p "$OUTPUT_DIRECTORY"
ln -sfn "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/output/stone_recovery_bc/latest_architecture_sweep"
export MINESTUDIO_BC_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY"
export MINESTUDIO_BC_TASK_BANK="$TASK_BANK"
export MINESTUDIO_BC_MAX_STEPS=6400
export MINESTUDIO_BC_VALIDATION_INTERVAL=200
export MINESTUDIO_BC_VALIDATION_BATCHES=20
export MINESTUDIO_BC_EARLY_STOPPING_PATIENCE=8
export MINESTUDIO_BC_SNAPSHOT_STEPS=800,1600,3200,4800,6400
export MINESTUDIO_BC_TELEGRAM=1

notify() {
    TELEGRAM_MESSAGE="$1" python - <<'PY' || true
import os

from telegram_training_helper import TelegramBot


TelegramBot(poll_updates=False, enable_terminal_commands=False, drop_pending_updates=False).send_message(text=os.environ["TELEGRAM_MESSAGE"])
PY
}

write_state() {
    local status=$1
    local variant=${2:-none}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"%s","variant":"%s","pid":%s,"output_directory":"%s","updated_at":"%s"}\n' "$status" "$RUN_ID" "$variant" "$$" "$OUTPUT_DIRECTORY" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed "${variant:-unknown}"
        notify "MineStudio architecture sweep failed
variant=${variant:-unknown}
output=$OUTPUT_DIRECTORY"
    fi
}

on_signal() {
    write_state interrupted "${variant:-unknown}"
    exit 143
}

trap on_exit EXIT
trap on_signal TERM INT

test -f "$TASK_BANK"
python -m pip freeze > "$OUTPUT_DIRECTORY/packages.txt"
nvidia-smi > "$OUTPUT_DIRECTORY/nvidia-smi.txt"
cp "$TASK_BANK" "$OUTPUT_DIRECTORY/gemini_task_bank.json"
cp "$PROJECT_DIRECTORY/output/stone_recovery_bc/dataset_v1/summary.json" "$OUTPUT_DIRECTORY/dataset_summary.json"
write_state running setup
notify "MineStudio 6400-step architecture sweep started
variants=${VARIANTS[*]}
validation_interval=200
early_stop_patience=8
output=$OUTPUT_DIRECTORY"

for variant in "${VARIANTS[@]}"; do
    write_state running "$variant"
    notify "MineStudio architecture training started
variant=$variant
max_steps=6400"
    MINESTUDIO_BC_VARIANT="$variant" python run/train_stone_recovery_bc.py 2>&1 | tee -a "$OUTPUT_DIRECTORY/${variant}.log"
done

variant=summarize
write_state running "$variant"
python run/summarize_stone_recovery_architecture_sweep.py > "$OUTPUT_DIRECTORY/architecture_report.stdout.json"
write_state completed none
notify "MineStudio architecture sweep completed
report=$OUTPUT_DIRECTORY/architecture_report.json"
