#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
TRAINING_DIRECTORY=$PROJECT_DIRECTORY/output/stone_recovery_bc/architecture_sweeps/20260917T213751Z
CHECKPOINT=$TRAINING_DIRECTORY/upper_lora_rank32/best_model
TASK_BANK=$TRAINING_DIRECTORY/gemini_task_bank.json
RUN_ID=${MINESTUDIO_SYSTEM0_RUN_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
OUTPUT_DIRECTORY=${MINESTUDIO_SYSTEM0_ABLATION_DIRECTORY:-$PROJECT_DIRECTORY/output/stone_acquisition/system0_router_ablations/$RUN_ID}
RUN_STATE=$OUTPUT_DIRECTORY/run_state.json
WORKERS=2

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
export PYTHONPATH="$PROJECT_DIRECTORY/run:$PYTHONPATH"
mkdir -p "$OUTPUT_DIRECTORY"
ln -sfn "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/output/stone_acquisition/latest_system0_router_ablation"
export MINESTUDIO_SYSTEM0_ABLATION_DIRECTORY="$OUTPUT_DIRECTORY"

notify() {
    TELEGRAM_MESSAGE="$1" python - <<'PY' || true
import os

from telegram_training_helper import TelegramBot


TelegramBot(poll_updates=False, enable_terminal_commands=False, drop_pending_updates=False).send_message(text=os.environ["TELEGRAM_MESSAGE"])
PY
}

write_state() {
    local status=$1
    local phase=${2:-none}
    local split=${3:-none}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"%s","phase":"%s","split":"%s","pid":%s,"output_directory":"%s","updated_at":"%s"}\n' "$status" "$RUN_ID" "$phase" "$split" "$$" "$OUTPUT_DIRECTORY" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed "${phase:-unknown}" "${split:-unknown}"
        notify "MineStudio System 0/router ablation failed
phase=${phase:-unknown}
split=${split:-unknown}
output=$OUTPUT_DIRECTORY"
    fi
}

trap on_exit EXIT

python -m pip freeze > "$OUTPUT_DIRECTORY/packages.txt"
nvidia-smi > "$OUTPUT_DIRECTORY/nvidia-smi.txt"
cp "$TASK_BANK" "$OUTPUT_DIRECTORY/gemini_task_bank.json"
write_state running preflight none
python -m unittest run.test_closed_loop_options_unit run.test_stone_task_router
python -m unittest discover -s tests -p 'test_system_zero*.py'

notify "MineStudio System 0/router paired ablation started
checkpoint=upper_lora_r32_step6400
normal=40
hazard=60
output=$OUTPUT_DIRECTORY"

for split in normal hazard; do
    phase=evaluation
    write_state running "$phase" "$split"
    if test "$split" = normal; then
        episodes=40
        seed=2026092301
        manifest=general_natural
    else
        episodes=60
        seed=2026092401
        manifest=hazard_stress
    fi
    MINESTUDIO_STONE_CHECKPOINT="$CHECKPOINT" \
    MINESTUDIO_STONE_EXPERIMENT="system0_ablation_${split}_repaired_v1" \
    MINESTUDIO_STONE_EPISODES="$episodes" \
    MINESTUDIO_STONE_WORKERS="$WORKERS" \
    MINESTUDIO_STONE_BASE_SEED="$seed" \
    MINESTUDIO_STONE_MANIFEST_KIND="$manifest" \
    MINESTUDIO_STONE_STAGE="System 0 and router paired ablation" \
    MINESTUDIO_STONE_EVENT_TASKS=1 \
    MINESTUDIO_STONE_TASK_BANK="$TASK_BANK" \
    python run/benchmark_stone_acquisition.py 2>&1 | tee "$OUTPUT_DIRECTORY/${split}_repaired.log"
done

phase=summarize
split=none
write_state running "$phase" "$split"
python run/summarize_system0_router_ablation.py > "$OUTPUT_DIRECTORY/system0_router_ablation_report.stdout.json"

phase=gemini_critic
write_state running "$phase" "$split"
MINESTUDIO_SCREEN_DIRECTORY="$OUTPUT_DIRECTORY" \
MINESTUDIO_CRITIC_POLICIES=repaired \
MINESTUDIO_CRITIC_SPLITS=normal,hazard \
MINESTUDIO_CRITIC_SUCCESS_SAMPLES=0 \
MINESTUDIO_CRITIC_EXPERIMENT_PREFIX=system0_ablation \
python run/criticize_stone_behavioral_screen.py 2>&1 | tee "$OUTPUT_DIRECTORY/gemini_critic.log"

critic_status=$(jq -r '.status' "$OUTPUT_DIRECTORY/gemini_trajectory_critic_summary.json")
if test "$critic_status" = blocked_billing; then
    write_state evaluation_complete_critic_blocked gemini_critic none
    notify "MineStudio System 0/router evaluation completed; Gemini critique paused for credits.
report=$OUTPUT_DIRECTORY/system0_router_ablation_report.json"
    exit 0
fi
write_state completed none none
notify "MineStudio System 0/router paired ablation completed
report=$OUTPUT_DIRECTORY/system0_router_ablation_report.json
critic=$OUTPUT_DIRECTORY/gemini_trajectory_critic_summary.json"
