#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
SCREEN_DIRECTORY=$PROJECT_DIRECTORY/output/stone_acquisition/behavioral_screens/20260919T111059Z
SYSTEM_ZERO_DIRECTORY=$PROJECT_DIRECTORY/output/system_zero
RUN_STATE=$SYSTEM_ZERO_DIRECTORY/run_state.json
LOG_PATH=$SYSTEM_ZERO_DIRECTORY/priority_one_smoke.log

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
mkdir -p "$SYSTEM_ZERO_DIRECTORY"

notify() {
    TELEGRAM_MESSAGE="$1" python - <<'PY' || true
import os

from telegram_training_helper import TelegramBot


TelegramBot(poll_updates=False, enable_terminal_commands=False, drop_pending_updates=False).send_message(text=os.environ["TELEGRAM_MESSAGE"])
PY
}

write_state() {
    local status=$1
    local phase=$2
    local reason=${3:-}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","phase":"%s","pid":%s,"reason":"%s","updated_at":"%s"}\n' "$status" "$phase" "$$" "$reason" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed "${phase:-unknown}" "System Zero smoke exited with code $exit_code"
        notify "MineStudio System Zero Priority 1 smoke failed. Log: $LOG_PATH"
    fi
}

trap on_exit EXIT

phase=behavioral_screen
write_state waiting behavioral_screen
while true; do
    if ! test -f "$SCREEN_DIRECTORY/run_state.json"; then
        write_state failed waiting "behavioral screen state missing"
        notify "MineStudio System Zero smoke could not start: behavioral screen state is missing."
        exit 1
    fi
    screen_status=$(jq -r '.status' "$SCREEN_DIRECTORY/run_state.json")
    complete_evaluations=true
    for split in normal hazard; do
        target=40
        if test "$split" = hazard; then target=60; fi
        for policy in vanilla upper_lora_r8_step800 upper_lora_r8_step1600 upper_lora_r8_step3200 upper_lora_r8_step4800 upper_lora_r8_step6400 upper_lora_r32_step6400 recurrent_upper_lora_step6400; do
            summary="$PROJECT_DIRECTORY/output/stone_acquisition/screen_${split}_${policy}_v1/summary.json"
            if ! test -f "$summary" || test "$(jq -r '.episodes_complete' "$summary")" -ne "$target"; then
                complete_evaluations=false
            fi
        done
    done
    if test "$screen_status" = completed || test "$screen_status" = evaluation_complete_critic_blocked || test "$complete_evaluations" = true && test -f "$SCREEN_DIRECTORY/behavioral_screen_report.json"; then
        break
    fi
    if test "$screen_status" = failed || test "$screen_status" = interrupted; then
        write_state blocked waiting "behavioral screen $screen_status"
        notify "MineStudio System Zero smoke blocked because the behavioral screen ended with status=$screen_status."
        exit 1
    fi
    sleep 60
done

phase=priority_one_smoke
write_state running priority_one_smoke
python run/system_zero_smoke_test.py 2>&1 | tee "$LOG_PATH"
write_state completed priority_one_smoke
notify "MineStudio System Zero Priority 1 smoke completed. Result: $SYSTEM_ZERO_DIRECTORY/priority_one_smoke.json"
