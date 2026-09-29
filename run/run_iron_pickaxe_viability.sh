#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
RUN_ID=${MINESTUDIO_IRON_RUN_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
OUTPUT_DIRECTORY=${MINESTUDIO_IRON_SUITE:-$PROJECT_DIRECTORY/output/iron_pickaxe_viability/$RUN_ID}
STATE_PATH=$OUTPUT_DIRECTORY/run_state.json
SMOKE_DIRECTORY=$PROJECT_DIRECTORY/output/iron_pickaxe_viability_smoke/$RUN_ID
phase=preflight
detail=environment

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
export PYTHONPATH="$PROJECT_DIRECTORY/run:$PYTHONPATH"
export MINESTUDIO_IRON_RUN_ID="$RUN_ID"
export MINESTUDIO_IRON_SUITE="$OUTPUT_DIRECTORY"
mkdir -p "$OUTPUT_DIRECTORY/logs" "$OUTPUT_DIRECTORY/code"
ln -sfn "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/output/iron_pickaxe_viability/latest"

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
        notify "MineStudio iron-pickaxe viability experiment failed
phase=$phase
detail=$detail
output=$OUTPUT_DIRECTORY"
    fi
}

trap on_exit EXIT
write_state running
python -m pip freeze > "$OUTPUT_DIRECTORY/packages.txt"
nvidia-smi > "$OUTPUT_DIRECTORY/nvidia-smi.txt"
free -h > "$OUTPUT_DIRECTORY/memory.txt"
df -hT / /mnt/hdd > "$OUTPUT_DIRECTORY/storage.txt"
cp run/benchmark_iron_pickaxe_viability.py run/prepare_iron_pickaxe_viability.py run/summarize_iron_pickaxe_viability.py run/test_iron_age_system0.py run/closed_loop_options.py run/run_iron_pickaxe_viability.sh minestudio/system_zero/skills.py "$OUTPUT_DIRECTORY/code/"
sha256sum "$OUTPUT_DIRECTORY"/code/* > "$OUTPUT_DIRECTORY/code_sha256.txt"
find checkpoints/steve_one_official -type f -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_DIRECTORY/vanilla_checkpoint_sha256.txt"
find output/stone_recovery_bc/architecture_sweeps/20260917T213751Z/upper_lora_rank32/best_model -type f -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_DIRECTORY/specialist_checkpoint_sha256.txt"
MINESTUDIO_IRON_EPISODES=24 MINESTUDIO_IRON_MAX_STEPS=12000 MINESTUDIO_IRON_MAX_DECISIONS=32 python run/prepare_iron_pickaxe_viability.py > "$OUTPUT_DIRECTORY/logs/manifest.log"

phase=system0_smoke
detail=furnace_smelt_iron_pickaxe
write_state running
MINESTUDIO_IRON_SYSTEM0_SMOKE="$OUTPUT_DIRECTORY/system0_smoke.json" python run/test_iron_age_system0.py 2>&1 | tee "$OUTPUT_DIRECTORY/logs/system0_smoke.log"
test "$(jq -r '.passed' "$OUTPUT_DIRECTORY/system0_smoke.json")" = true

phase=paired_smoke
detail=one_seed_each_arm
write_state running
mkdir -p "$SMOKE_DIRECTORY"
MINESTUDIO_IRON_SUITE="$SMOKE_DIRECTORY" MINESTUDIO_IRON_EPISODES=1 MINESTUDIO_IRON_MAX_STEPS=1200 MINESTUDIO_IRON_MAX_DECISIONS=3 python run/prepare_iron_pickaxe_viability.py > "$OUTPUT_DIRECTORY/logs/smoke_manifest.log"
smoke_pids=()
for arm in full no_specialist no_system2; do
    MINESTUDIO_IRON_SUITE="$SMOKE_DIRECTORY" MINESTUDIO_IRON_ARM="$arm" MINESTUDIO_IRON_EPISODES=1 MINESTUDIO_IRON_MAX_STEPS=1200 MINESTUDIO_IRON_MAX_DECISIONS=3 python run/benchmark_iron_pickaxe_viability.py > "$OUTPUT_DIRECTORY/logs/smoke_${arm}.log" 2>&1 &
    smoke_pids+=("$!")
done
for pid in "${smoke_pids[@]}"; do
    wait "$pid"
done
for arm in full no_specialist no_system2; do
    test "$(jq -r '.episodes_valid' "$SMOKE_DIRECTORY/$arm/summary.json")" = 1
done

phase=evaluation
detail=72_paired_episodes
write_state running
notify "MineStudio final viability evaluation launched
objective=build iron pickaxe
arms=FULL, NO SPECIALIST, NO SYSTEM 2
paired_seeds=24
episodes=72
training=false
output=$OUTPUT_DIRECTORY"
pids=()
for arm in full no_specialist no_system2; do
    MINESTUDIO_IRON_ARM="$arm" MINESTUDIO_IRON_EPISODES=24 MINESTUDIO_IRON_MAX_STEPS=12000 MINESTUDIO_IRON_MAX_DECISIONS=32 python run/benchmark_iron_pickaxe_viability.py > "$OUTPUT_DIRECTORY/logs/${arm}.log" 2>&1 &
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
