#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
TRAINING_DIRECTORY=$PROJECT_DIRECTORY/output/stone_recovery_bc/architecture_sweeps/20260917T213751Z
CHECKPOINT=$TRAINING_DIRECTORY/upper_lora_rank32/best_model
RUN_ID=${MINESTUDIO_MICRO_RUN_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
SUITE_DIRECTORY=${MINESTUDIO_MICRO_SUITE_DIRECTORY:-$PROJECT_DIRECTORY/output/system1_recovery_micro/$RUN_ID}
RUN_STATE=$SUITE_DIRECTORY/run_state.json
TASKS=(EXIT_WATER CLIMB_SHORE REACQUIRE_STONE ESCAPE_HOLE AVOID_DIGGING_TRAP RECOVER_CAMERA AVOID_WATER)
BASE_SEED=2026092501
WORKERS=${MINESTUDIO_MICRO_WORKERS:-2}
EPISODES=50
GPU_RETRY_SECONDS=${MINESTUDIO_MICRO_GPU_RETRY_SECONDS:-30}

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
export PYTHONPATH="$PROJECT_DIRECTORY/run:$PYTHONPATH"
export MINESTUDIO_MICRO_SUITE_DIRECTORY="$SUITE_DIRECTORY"
mkdir -p "$SUITE_DIRECTORY/code" "$SUITE_DIRECTORY/logs" "$SUITE_DIRECTORY/smoke"
ln -sfn "$SUITE_DIRECTORY" "$PROJECT_DIRECTORY/output/system1_recovery_micro/latest"

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
    local task=${3:-none}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"%s","phase":"%s","task":"%s","pid":%s,"suite_directory":"%s","updated_at":"%s"}\n' "$status" "$RUN_ID" "$phase" "$task" "$$" "$SUITE_DIRECTORY" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

wait_for_external_gpu_job() {
    while pgrep -af '[v]llm serve' >/dev/null; do
        write_state waiting_gpu evaluation "$task"
        sleep "$GPU_RETRY_SECONDS"
    done
}

run_evaluation_task() {
    local current_task=$1
    local task_output=$SUITE_DIRECTORY/${current_task,,}
    local task_log=$SUITE_DIRECTORY/logs/${current_task,,}.log
    local command_status

    while true; do
        wait_for_external_gpu_job
        write_state running evaluation "$current_task"
        set +e
        MINESTUDIO_MICRO_CHECKPOINT="$CHECKPOINT" \
        MINESTUDIO_MICRO_OUTPUT="$task_output" \
        MINESTUDIO_MICRO_TASK="$current_task" \
        MINESTUDIO_MICRO_EPISODES="$EPISODES" \
        MINESTUDIO_MICRO_WORKERS="$WORKERS" \
        MINESTUDIO_MICRO_BASE_SEED="$BASE_SEED" \
        python run/benchmark_system1_recovery_micro.py 2>&1 | tee -a "$task_log"
        command_status=${PIPESTATUS[0]}
        set -e

        if test "$command_status" -eq 0; then
            return 0
        fi

        if tail -n 8 "$task_output/worker_errors.jsonl" 2>/dev/null | grep -qiE 'CUDA error: out of memory|torch\.OutOfMemoryError'; then
            write_state waiting_gpu evaluation "$current_task"
            sleep "$GPU_RETRY_SECONDS"
            continue
        fi

        return "$command_status"
    done
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed "${phase:-unknown}" "${task:-unknown}"
        notify "MineStudio System 1 recovery micro-benchmarks failed
phase=${phase:-unknown}
task=${task:-unknown}
output=$SUITE_DIRECTORY"
    fi
}

trap on_exit EXIT

python -m pip freeze > "$SUITE_DIRECTORY/packages.txt"
nvidia-smi > "$SUITE_DIRECTORY/nvidia-smi.txt"
cp run/benchmark_system1_recovery_micro.py run/test_system1_recovery_micro.py run/criticize_system1_recovery_micro.py run/summarize_system1_recovery_micro.py run/run_system1_recovery_micro.sh run/closed_loop_options.py run/stone_task_router.py "$SUITE_DIRECTORY/code/"
sha256sum "$SUITE_DIRECTORY"/code/* > "$SUITE_DIRECTORY/code_sha256.txt"
find "$CHECKPOINT" -type f -print0 | sort -z | xargs -0 sha256sum > "$SUITE_DIRECTORY/checkpoint_sha256.txt"

phase=preflight
task=none
write_state running "$phase" "$task"
python -m unittest run.test_system1_recovery_micro

for task in "${TASKS[@]}"; do
    phase=smoke
    write_state running "$phase" "$task"
    smoke_output=$SUITE_DIRECTORY/smoke/${task,,}
    MINESTUDIO_MICRO_CHECKPOINT="$CHECKPOINT" \
    MINESTUDIO_MICRO_OUTPUT="$smoke_output" \
    MINESTUDIO_MICRO_TASK="$task" \
    MINESTUDIO_MICRO_EPISODES=1 \
    MINESTUDIO_MICRO_WORKERS=1 \
    MINESTUDIO_MICRO_BASE_SEED="$BASE_SEED" \
    python run/benchmark_system1_recovery_micro.py 2>&1 | tee "$SUITE_DIRECTORY/logs/smoke_${task,,}.log"
done

notify "MineStudio System 1 recovery micro-benchmarks started
checkpoint=upper_lora_r32_step6400
router=old
pickup=old
tasks=7
episodes=50_each
training=false
output=$SUITE_DIRECTORY"

for task in "${TASKS[@]}"; do
    phase=evaluation
    run_evaluation_task "$task"
done

phase=summarize_environment
task=none
write_state running "$phase" "$task"
python run/summarize_system1_recovery_micro.py > "$SUITE_DIRECTORY/environment_report.stdout.json"

phase=gemini_critic
write_state running "$phase" "$task"
python run/criticize_system1_recovery_micro.py 2>&1 | tee "$SUITE_DIRECTORY/logs/gemini_critic.log"
critic_status=$(jq -r '.status' "$SUITE_DIRECTORY/gemini_critic_summary.json")
if test "$critic_status" = blocked_billing; then
    phase=gemini_critic
    write_state evaluation_complete_critic_blocked "$phase" none
    notify "MineStudio System 1 micro-benchmark evaluation completed; Gemini critique paused for credits.
output=$SUITE_DIRECTORY"
    exit 0
fi

phase=summarize_final
write_state running "$phase" none
python run/summarize_system1_recovery_micro.py > "$SUITE_DIRECTORY/system1_recovery_micro_report.stdout.json"
phase=none
write_state completed "$phase" none
notify "MineStudio System 1 recovery micro-benchmarks completed
report=$SUITE_DIRECTORY/system1_recovery_micro_report.json
critic=$SUITE_DIRECTORY/gemini_critic_summary.json"
