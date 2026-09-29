#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
TRAINING_DIRECTORY=$PROJECT_DIRECTORY/output/stone_recovery_bc/architecture_sweeps/20260917T213751Z
CHECKPOINT=$TRAINING_DIRECTORY/upper_lora_rank32/best_model
TASK_BANK=$TRAINING_DIRECTORY/gemini_task_bank.json
RUN_ID=${MINESTUDIO_SYSTEM0_ISOLATION_RUN_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
OUTPUT_DIRECTORY=${MINESTUDIO_SYSTEM0_ISOLATION_DIRECTORY:-$PROJECT_DIRECTORY/output/stone_acquisition/system0_isolations/$RUN_ID}
RUN_STATE=$OUTPUT_DIRECTORY/run_state.json
WORKERS=2

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
export PYTHONPATH="$PROJECT_DIRECTORY/run:$PYTHONPATH"
mkdir -p "$OUTPUT_DIRECTORY/code"
ln -sfn "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/output/stone_acquisition/latest_system0_isolation"
export MINESTUDIO_SYSTEM0_ISOLATION_DIRECTORY="$OUTPUT_DIRECTORY"

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
    local configuration=${3:-none}
    local split=${4:-none}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"%s","phase":"%s","configuration":"%s","split":"%s","pid":%s,"output_directory":"%s","updated_at":"%s"}\n' "$status" "$RUN_ID" "$phase" "$configuration" "$split" "$$" "$OUTPUT_DIRECTORY" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed "${phase:-unknown}" "${configuration:-unknown}" "${split:-unknown}"
        notify "MineStudio System 0 isolation failed
phase=${phase:-unknown}
configuration=${configuration:-unknown}
split=${split:-unknown}
output=$OUTPUT_DIRECTORY"
    fi
}

trap on_exit EXIT

python -m pip freeze > "$OUTPUT_DIRECTORY/packages.txt"
nvidia-smi > "$OUTPUT_DIRECTORY/nvidia-smi.txt"
cp "$TASK_BANK" "$OUTPUT_DIRECTORY/gemini_task_bank.json"
cp run/benchmark_stone_acquisition.py run/closed_loop_options.py run/stone_task_router.py run/summarize_system0_isolation.py run/verify_system0_isolation_setup.py "$OUTPUT_DIRECTORY/code/"
sha256sum "$OUTPUT_DIRECTORY"/code/* > "$OUTPUT_DIRECTORY/code_sha256.txt"
write_state running preflight none none
python -m unittest run.test_closed_loop_options_unit run.test_stone_task_router
python -m unittest discover -s tests -p 'test_system_zero*.py'
python run/verify_system0_isolation_setup.py > "$OUTPUT_DIRECTORY/setup_verification.json"

notify "MineStudio System 0 pickup/router isolation started
checkpoint=upper_lora_r32_step6400
configurations=old_router_hardened_pickup,new_router_old_pickup
episodes=100_each
output=$OUTPUT_DIRECTORY"

for configuration in old_router_hardened_pickup new_router_old_pickup; do
    if test "$configuration" = old_router_hardened_pickup; then
        router=old
        pickup=hardened
    else
        router=new
        pickup=old
    fi
    for split in normal hazard; do
        phase=evaluation
        write_state running "$phase" "$configuration" "$split"
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
        MINESTUDIO_STONE_EXPERIMENT="system0_isolation_${split}_${configuration}_v1" \
        MINESTUDIO_STONE_EPISODES="$episodes" \
        MINESTUDIO_STONE_WORKERS="$WORKERS" \
        MINESTUDIO_STONE_BASE_SEED="$seed" \
        MINESTUDIO_STONE_MANIFEST_KIND="$manifest" \
        MINESTUDIO_STONE_STAGE="System 0 pickup/router isolation" \
        MINESTUDIO_STONE_EVENT_TASKS=1 \
        MINESTUDIO_STONE_ROUTER="$router" \
        MINESTUDIO_STONE_PICKUP="$pickup" \
        MINESTUDIO_STONE_TASK_BANK="$TASK_BANK" \
        python run/benchmark_stone_acquisition.py 2>&1 | tee "$OUTPUT_DIRECTORY/${configuration}_${split}.log"
    done
done

phase=summarize
configuration=none
split=none
write_state running "$phase" "$configuration" "$split"
python run/summarize_system0_isolation.py > "$OUTPUT_DIRECTORY/system0_isolation_report.stdout.json"

phase=gemini_critic
write_state running "$phase" "$configuration" "$split"
MINESTUDIO_SCREEN_DIRECTORY="$OUTPUT_DIRECTORY" \
MINESTUDIO_CRITIC_POLICIES=old_router_hardened_pickup,new_router_old_pickup \
MINESTUDIO_CRITIC_SPLITS=normal,hazard \
MINESTUDIO_CRITIC_SUCCESS_SAMPLES=0 \
MINESTUDIO_CRITIC_EXPERIMENT_PREFIX=system0_isolation \
python run/criticize_stone_behavioral_screen.py 2>&1 | tee "$OUTPUT_DIRECTORY/gemini_critic.log"

critic_status=$(jq -r '.status' "$OUTPUT_DIRECTORY/gemini_trajectory_critic_summary.json")
if test "$critic_status" = blocked_billing; then
    write_state evaluation_complete_critic_blocked gemini_critic none none
    notify "MineStudio System 0 isolation evaluation completed; Gemini critique paused for credits.
report=$OUTPUT_DIRECTORY/system0_isolation_report.json"
    exit 0
fi
write_state completed none none none
notify "MineStudio System 0 pickup/router isolation completed
report=$OUTPUT_DIRECTORY/system0_isolation_report.json
critic=$OUTPUT_DIRECTORY/gemini_trajectory_critic_summary.json"
