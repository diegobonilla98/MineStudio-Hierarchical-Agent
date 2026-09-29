#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
RUN_ID=${MINESTUDIO_PPO_RUN_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
OUTPUT_DIRECTORY=${MINESTUDIO_PPO_OUTPUT:-$PROJECT_DIRECTORY/output/system1_recovery_ppo/$RUN_ID}
RUN_STATE=$OUTPUT_DIRECTORY/pipeline_state.json
REFERENCE_CHECKPOINT=$PROJECT_DIRECTORY/output/stone_recovery_bc/architecture_sweeps/20260917T213751Z/upper_lora_rank32/best_model
TASK_BANK=$PROJECT_DIRECTORY/output/stone_recovery_bc/architecture_sweeps/20260917T213751Z/gemini_task_bank.json
BASELINE_MICRO=$PROJECT_DIRECTORY/output/system1_recovery_micro/20260920T075136Z
TASKS=(EXIT_WATER CLIMB_SHORE REACQUIRE_STONE ESCAPE_HOLE AVOID_DIGGING_TRAP RECOVER_CAMERA AVOID_WATER)
GPU_RETRY_SECONDS=30

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
export PYTHONPATH="$PROJECT_DIRECTORY/run:$PYTHONPATH"
export MINESTUDIO_PPO_RUN_ID="$RUN_ID"
export MINESTUDIO_PPO_OUTPUT="$OUTPUT_DIRECTORY"
export MINESTUDIO_BC_TASK_BANK="$TASK_BANK"
mkdir -p "$OUTPUT_DIRECTORY/code" "$OUTPUT_DIRECTORY/logs" "$OUTPUT_DIRECTORY/final_evaluation"
ln -sfn "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/output/system1_recovery_ppo/latest"

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
    local detail=${3:-none}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"%s","phase":"%s","detail":"%s","pid":%s,"output_directory":"%s","updated_at":"%s"}\n' "$status" "$RUN_ID" "$phase" "$detail" "$$" "$OUTPUT_DIRECTORY" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

wait_for_gpu_slot() {
    while test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"; do
        write_state waiting_gpu "$phase" "$detail"
        sleep "$GPU_RETRY_SECONDS"
    done
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed "${phase:-unknown}" "${detail:-unknown}"
        notify "MineStudio targeted System 1 PPO failed
phase=${phase:-unknown}
detail=${detail:-unknown}
output=$OUTPUT_DIRECTORY"
    fi
}

trap on_exit EXIT

phase=preflight
detail=environment
write_state running "$phase" "$detail"
wait_for_gpu_slot
python -m pip freeze > "$OUTPUT_DIRECTORY/packages.txt"
nvidia-smi > "$OUTPUT_DIRECTORY/nvidia-smi.txt"
free -h > "$OUTPUT_DIRECTORY/memory.txt"
df -hT / /mnt/hdd > "$OUTPUT_DIRECTORY/storage.txt"
cp run/train_system1_recovery_ppo.py run/test_train_system1_recovery_ppo.py run/test_malmo_comms_transport.py run/finalize_system1_recovery_ppo.py run/summarize_system1_recovery_ppo.py run/plot_system1_recovery_ppo.py run/benchmark_system1_recovery_micro.py run/criticize_system1_recovery_micro.py run/summarize_system1_recovery_micro.py run/benchmark_stone_acquisition.py run/closed_loop_options.py run/stone_task_router.py run/run_system1_recovery_ppo.sh minestudio/simulator/minerl/env/comms.py "$OUTPUT_DIRECTORY/code/"
sha256sum "$OUTPUT_DIRECTORY"/code/* > "$OUTPUT_DIRECTORY/code_sha256.txt"
find "$REFERENCE_CHECKPOINT" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_DIRECTORY/reference_checkpoint_sha256.txt"
python -m unittest run.test_train_system1_recovery_ppo

phase=smoke
detail=forward_backward_checkpoint
write_state running "$phase" "$detail"
wait_for_gpu_slot
SMOKE_DIRECTORY=$PROJECT_DIRECTORY/output/system1_recovery_ppo_smoke/$RUN_ID
if test ! -f "$SMOKE_DIRECTORY/run_state.json" || test "$(jq -r '.status' "$SMOKE_DIRECTORY/run_state.json")" != completed || test ! -f "$SMOKE_DIRECTORY/last_training_state.pt" || test ! -f "$SMOKE_DIRECTORY/smoke_model/model.safetensors"; then
    MINESTUDIO_PPO_SMOKE=1 \
    MINESTUDIO_PPO_TELEGRAM=0 \
    MINESTUDIO_PPO_OUTPUT="$SMOKE_DIRECTORY" \
    MINESTUDIO_PPO_RUN_ID="$RUN_ID-smoke" \
    python run/train_system1_recovery_ppo.py 2>&1 | tee "$OUTPUT_DIRECTORY/logs/smoke.log"
fi
test "$(jq -r '.status' "$SMOKE_DIRECTORY/run_state.json")" = completed
test -f "$SMOKE_DIRECTORY/last_training_state.pt"
test -f "$SMOKE_DIRECTORY/smoke_model/model.safetensors"

phase=training
detail=targeted_ppo
write_state running "$phase" "$detail"
wait_for_gpu_slot
notify "MineStudio targeted System 1 PPO launched
reference=upper_lora_r32_step6400
router=old
pickup=old
tasks=6
training=true
output=$OUTPUT_DIRECTORY"
if test ! -f "$OUTPUT_DIRECTORY/summary.json" || test "$(jq -r '.status' "$OUTPUT_DIRECTORY/summary.json")" != completed; then
    MINESTUDIO_PPO_SMOKE=0 \
    MINESTUDIO_PPO_TELEGRAM=1 \
    python run/train_system1_recovery_ppo.py 2>&1 | tee -a "$OUTPUT_DIRECTORY/logs/training.log"
fi
CANDIDATE_CHECKPOINT=$(jq -r '.best.checkpoint_directory' "$OUTPUT_DIRECTORY/summary.json")
test -f "$CANDIDATE_CHECKPOINT/model.safetensors"
python run/plot_system1_recovery_ppo.py

phase=final_micro
detail=candidate
write_state running "$phase" "$detail"
wait_for_gpu_slot
CANDIDATE_MICRO=$OUTPUT_DIRECTORY/final_evaluation/micro_candidate
mkdir -p "$CANDIDATE_MICRO"
for task in "${TASKS[@]}"; do
    detail=$task
    write_state running "$phase" "$detail"
    MINESTUDIO_MICRO_CHECKPOINT="$CANDIDATE_CHECKPOINT" \
    MINESTUDIO_MICRO_CHECKPOINT_ROLE="targeted PPO candidate" \
    MINESTUDIO_MICRO_OUTPUT="$CANDIDATE_MICRO/${task,,}" \
    MINESTUDIO_MICRO_TASK="$task" \
    MINESTUDIO_MICRO_EPISODES=50 \
    MINESTUDIO_MICRO_WORKERS=1 \
    MINESTUDIO_MICRO_BASE_SEED=2026092501 \
    python run/benchmark_system1_recovery_micro.py 2>&1 | tee -a "$OUTPUT_DIRECTORY/logs/final_micro_${task,,}.log"
done

phase=final_stone
detail=normal
write_state running "$phase" "$detail"
wait_for_gpu_slot
MINESTUDIO_STONE_CHECKPOINT="$CANDIDATE_CHECKPOINT" \
MINESTUDIO_STONE_EXPERIMENT="ppo_${RUN_ID}_normal_candidate" \
MINESTUDIO_STONE_EPISODES=40 \
MINESTUDIO_STONE_WORKERS=1 \
MINESTUDIO_STONE_BASE_SEED=2026092301 \
MINESTUDIO_STONE_MANIFEST_KIND=general_natural \
MINESTUDIO_STONE_STAGE="Targeted PPO final paired evaluation" \
MINESTUDIO_STONE_EVENT_TASKS=1 \
MINESTUDIO_STONE_TASK_BANK="$TASK_BANK" \
MINESTUDIO_STONE_ROUTER=old \
MINESTUDIO_STONE_PICKUP=old \
python run/benchmark_stone_acquisition.py 2>&1 | tee -a "$OUTPUT_DIRECTORY/logs/final_stone_normal.log"

phase=final_stone
detail=hazard
write_state running "$phase" "$detail"
MINESTUDIO_STONE_CHECKPOINT="$CANDIDATE_CHECKPOINT" \
MINESTUDIO_STONE_EXPERIMENT="ppo_${RUN_ID}_hazard_candidate" \
MINESTUDIO_STONE_EPISODES=60 \
MINESTUDIO_STONE_WORKERS=1 \
MINESTUDIO_STONE_BASE_SEED=2026092401 \
MINESTUDIO_STONE_MANIFEST_KIND=hazard_stress \
MINESTUDIO_STONE_STAGE="Targeted PPO final paired evaluation" \
MINESTUDIO_STONE_EVENT_TASKS=1 \
MINESTUDIO_STONE_TASK_BANK="$TASK_BANK" \
MINESTUDIO_STONE_ROUTER=old \
MINESTUDIO_STONE_PICKUP=old \
python run/benchmark_stone_acquisition.py 2>&1 | tee -a "$OUTPUT_DIRECTORY/logs/final_stone_hazard.log"

phase=gemini_critic
detail=candidate_failures
write_state running "$phase" "$detail"
MINESTUDIO_MICRO_SUITE_DIRECTORY="$CANDIDATE_MICRO" python run/criticize_system1_recovery_micro.py 2>&1 | tee -a "$OUTPUT_DIRECTORY/logs/gemini_critic.log"
CRITIC_STATUS=$(jq -r '.status' "$CANDIDATE_MICRO/gemini_critic_summary.json")
if test "$CRITIC_STATUS" = blocked_billing; then
    write_state evaluation_complete_critic_blocked "$phase" "$detail"
    notify "MineStudio PPO training and final evaluations completed; Gemini critique paused for credits.
output=$OUTPUT_DIRECTORY"
    exit 0
fi

phase=summarize
detail=final_report
write_state running "$phase" "$detail"
MINESTUDIO_MICRO_SUITE_DIRECTORY="$CANDIDATE_MICRO" python run/summarize_system1_recovery_micro.py > "$CANDIDATE_MICRO/system1_recovery_micro_report.stdout.json"
python run/summarize_system1_recovery_ppo.py > "$OUTPUT_DIRECTORY/final_report.stdout.json"
phase=none
detail=none
write_state completed "$phase" "$detail"
notify "MineStudio targeted System 1 PPO experiment completed
report=$OUTPUT_DIRECTORY/final_report.json
training=$OUTPUT_DIRECTORY/summary.json"
