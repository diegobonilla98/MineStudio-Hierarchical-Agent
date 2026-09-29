#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
TRAINING_DIRECTORY=$PROJECT_DIRECTORY/output/stone_recovery_bc/runs/20260916T0952Z
OUTPUT_ROOT=$PROJECT_DIRECTORY/output/stone_acquisition
SUITE_ID=${MINESTUDIO_EVAL_SUITE_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
SUITE_DIRECTORY=${MINESTUDIO_EVAL_SUITE_DIRECTORY:-$OUTPUT_ROOT/evaluation_suites/$SUITE_ID}
FROZEN_SEED=2026091501
HELDOUT_SEED=2026091701
EPISODES=1000
WORKERS=2
RUN_STATE=$SUITE_DIRECTORY/run_state.json

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
mkdir -p "$SUITE_DIRECTORY"
ln -sfn "$SUITE_DIRECTORY" "$OUTPUT_ROOT/latest_evaluation_suite"
export MINESTUDIO_EVAL_SUITE_DIRECTORY="$SUITE_DIRECTORY"

notify() {
    TELEGRAM_MESSAGE="$1" python - <<'PY' || true
import os

from telegram_training_helper import TelegramBot


TelegramBot(poll_updates=False, enable_terminal_commands=False, drop_pending_updates=False).send_message(text=os.environ["TELEGRAM_MESSAGE"])
PY
}

write_state() {
    local status=$1
    local split=${2:-none}
    local variant=${3:-none}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","suite_id":"%s","split":"%s","variant":"%s","pid":%s,"updated_at":"%s"}\n' "$status" "$SUITE_ID" "$split" "$variant" "$$" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

checkpoint_for() {
    local variant=$1
    if test "$variant" = vanilla; then
        echo "$PROJECT_DIRECTORY/checkpoints/steve_one_official"
    else
        echo "$TRAINING_DIRECTORY/$variant/best_model"
    fi
}

run_evaluation() {
    local split=$1
    local variant=$2
    local base_seed=$3
    local experiment_name=stage1_${split}_${variant}_v1
    local experiment_directory=$OUTPUT_ROOT/$experiment_name
    local checkpoint
    checkpoint=$(checkpoint_for "$variant")
    mkdir -p "$experiment_directory"
    if test "$split" = frozen && ! test -f "$experiment_directory/manifest.jsonl"; then
        cp "$PROJECT_DIRECTORY/output/stone_recovery_bc/dataset_v1/frozen_manifest.jsonl" "$experiment_directory/manifest.jsonl"
    fi
    write_state running "$split" "$variant"
    notify "MineStudio evaluation started
split=$split
variant=$variant
episodes=$EPISODES
output=$experiment_directory"
    MINESTUDIO_STONE_CHECKPOINT="$checkpoint" \
    MINESTUDIO_STONE_EXPERIMENT="$experiment_name" \
    MINESTUDIO_STONE_EPISODES="$EPISODES" \
    MINESTUDIO_STONE_WORKERS="$WORKERS" \
    MINESTUDIO_STONE_BASE_SEED="$base_seed" \
    MINESTUDIO_STONE_STAGE="Stage 1 BC $split evaluation" \
    python run/benchmark_stone_acquisition.py 2>&1 | tee -a "$SUITE_DIRECTORY/${split}_${variant}.log"
    local success_rate
    success_rate=$(jq -r '.overall.success_rate' "$experiment_directory/summary.json")
    notify "MineStudio evaluation completed
split=$split
variant=$variant
success_rate=$success_rate
output=$experiment_directory"
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed "${split:-unknown}" "${variant:-unknown}"
        notify "MineStudio evaluation suite failed
split=${split:-unknown}
variant=${variant:-unknown}
output=$SUITE_DIRECTORY"
    fi
}

on_signal() {
    write_state interrupted "${split:-unknown}" "${variant:-unknown}"
    exit 143
}

trap on_exit EXIT
trap on_signal TERM INT

python -m pip freeze > "$SUITE_DIRECTORY/packages.txt"
nvidia-smi > "$SUITE_DIRECTORY/nvidia-smi.txt"
write_state running setup none
notify "MineStudio full stone recovery evaluation suite started
suite=$SUITE_ID
frozen=3x1000
heldout=4x1000
output=$SUITE_DIRECTORY"

for variant in action_head action_head_recurrent upper_lora; do
    run_evaluation frozen "$variant" "$FROZEN_SEED"
done

for variant in vanilla action_head action_head_recurrent upper_lora; do
    run_evaluation heldout "$variant" "$HELDOUT_SEED"
done

python run/summarize_stone_recovery_evaluations.py > "$SUITE_DIRECTORY/evaluation_report.stdout.json"
write_state completed none none
notify "MineStudio full stone recovery evaluation suite completed
report=$SUITE_DIRECTORY/evaluation_report.json"
