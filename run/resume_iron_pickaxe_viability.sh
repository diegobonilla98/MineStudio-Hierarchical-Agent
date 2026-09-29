#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
RUN_ID=20260921T134140Z
OUTPUT_DIRECTORY=$PROJECT_DIRECTORY/output/iron_pickaxe_viability/$RUN_ID
STATE_PATH=$OUTPUT_DIRECTORY/run_state.json
phase=resume_evaluation
detail=full_and_no_specialist

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
export PYTHONPATH="$PROJECT_DIRECTORY/run:$PYTHONPATH"
export MINESTUDIO_IRON_RUN_ID="$RUN_ID"
export MINESTUDIO_IRON_SUITE="$OUTPUT_DIRECTORY"

notify() {
    TELEGRAM_MESSAGE="$1" python - <<'PY' || true
import os

from telegram_training_helper import TelegramBot


TelegramBot(poll_updates=False, enable_terminal_commands=False, drop_pending_updates=False).send_message(text=os.environ["TELEGRAM_MESSAGE"])
PY
}

write_state() {
    local status=$1
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"%s","phase":"%s","detail":"%s","pid":%s,"output_directory":"%s","updated_at":"%s"}\n' "$status" "$RUN_ID" "$phase" "$detail" "$$" "$OUTPUT_DIRECTORY" "$timestamp" > "$STATE_PATH.tmp"
    mv "$STATE_PATH.tmp" "$STATE_PATH"
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed
        notify "MineStudio iron-pickaxe viability resume failed
phase=$phase
detail=$detail
output=$OUTPUT_DIRECTORY"
    fi
}

trap on_exit EXIT
write_state running
cp run/benchmark_iron_pickaxe_viability.py run/resume_iron_pickaxe_viability.sh "$OUTPUT_DIRECTORY/code/"
sha256sum "$OUTPUT_DIRECTORY/code/benchmark_iron_pickaxe_viability.py" "$OUTPUT_DIRECTORY/code/resume_iron_pickaxe_viability.sh" > "$OUTPUT_DIRECTORY/resume_code_sha256.txt"
notify "MineStudio iron-pickaxe viability evaluation resumed
remaining_arms=FULL, NO SPECIALIST
completed_arm=NO SYSTEM 2
output=$OUTPUT_DIRECTORY"

pids=()
for arm in full no_specialist; do
    completed=$(jq -s '[.[] | select(.status == "complete")] | length' "$OUTPUT_DIRECTORY/$arm/results.jsonl")
    if test "$completed" -ge 24; then
        continue
    fi
    MINESTUDIO_IRON_ARM="$arm" MINESTUDIO_IRON_EPISODES=24 MINESTUDIO_IRON_MAX_STEPS=12000 MINESTUDIO_IRON_MAX_DECISIONS=32 python run/benchmark_iron_pickaxe_viability.py >> "$OUTPUT_DIRECTORY/logs/${arm}.log" 2>&1 &
    echo "$!" > "$OUTPUT_DIRECTORY/${arm}.pid"
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

phase=summarize
detail=paired_statistics
write_state running
python run/summarize_iron_pickaxe_viability.py 2>&1 | tee "$OUTPUT_DIRECTORY/logs/summarize.log"

phase=none
detail=none
write_state completed
notify "MineStudio final viability evaluation completed
report=$OUTPUT_DIRECTORY/final_report.txt
json=$OUTPUT_DIRECTORY/final_report.json"
