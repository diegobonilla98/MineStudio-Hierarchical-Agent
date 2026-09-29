#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
SOURCE_TRAINING_DIRECTORY=$PROJECT_DIRECTORY/output/stone_recovery_bc/runs/20260916T0952Z
RUN_ID=${MINESTUDIO_CONFIRMATION_RUN_ID:-$(date --utc +%Y%m%dT%H%M%SZ)}
TRAINING_DIRECTORY=$PROJECT_DIRECTORY/output/stone_recovery_bc/runs/${RUN_ID}_lora800
SUITE_DIRECTORY=${MINESTUDIO_CONFIRMATION_SUITE_DIRECTORY:-$PROJECT_DIRECTORY/output/stone_acquisition/confirmation_suites/$RUN_ID}
RUN_STATE=$SUITE_DIRECTORY/run_state.json
GENERAL_EPISODES=1000
STRESS_EPISODES=600
GENERAL_SEED=2026100101
STRESS_SEED=2026101101
WORKERS=2

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
mkdir -p "$SUITE_DIRECTORY"
ln -sfn "$SUITE_DIRECTORY" "$PROJECT_DIRECTORY/output/stone_acquisition/latest_confirmation_suite"
export MINESTUDIO_CONFIRMATION_SUITE_DIRECTORY="$SUITE_DIRECTORY"

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
    local policy=${4:-none}
    local timestamp
    timestamp=$(date --iso-8601=seconds)
    printf '{"status":"%s","run_id":"%s","phase":"%s","split":"%s","policy":"%s","pid":%s,"training_directory":"%s","updated_at":"%s"}\n' "$status" "$RUN_ID" "$phase" "$split" "$policy" "$$" "$TRAINING_DIRECTORY" "$timestamp" > "$RUN_STATE.tmp"
    mv "$RUN_STATE.tmp" "$RUN_STATE"
}

on_exit() {
    local exit_code=$?
    if test "$exit_code" -ne 0; then
        write_state failed "${phase:-unknown}" "${split:-unknown}" "${policy:-unknown}"
        notify "MineStudio LoRA-800 confirmation failed
phase=${phase:-unknown}
split=${split:-unknown}
policy=${policy:-unknown}
suite=$SUITE_DIRECTORY"
    fi
}

on_signal() {
    write_state interrupted "${phase:-unknown}" "${split:-unknown}" "${policy:-unknown}"
    exit 143
}

trap on_exit EXIT
trap on_signal TERM INT

prepare_training() {
    phase=prepare_training
    write_state running "$phase" none none
    if ! test -f "$TRAINING_DIRECTORY/upper_lora/last_training_state.pt"; then
        mkdir -p "$TRAINING_DIRECTORY"
        cp -a "$SOURCE_TRAINING_DIRECTORY/upper_lora" "$TRAINING_DIRECTORY/upper_lora"
        mkdir -p "$TRAINING_DIRECTORY/upper_lora/snapshots"
        cp "$SOURCE_TRAINING_DIRECTORY/upper_lora/best_trainable.pt" "$TRAINING_DIRECTORY/upper_lora/snapshots/step_400_trainable.pt"
        cp "$SOURCE_TRAINING_DIRECTORY/upper_lora/last_training_state.pt" "$TRAINING_DIRECTORY/upper_lora/snapshots/step_400_training_state.pt"
        printf '%s\n' "$SOURCE_TRAINING_DIRECTORY/upper_lora/best_model" > "$TRAINING_DIRECTORY/upper_lora/snapshots/step_400_model.path"
    fi
}

train_lora800() {
    phase=train_lora800
    write_state running "$phase" none lora800
    notify "MineStudio Upper LoRA continuation started
steps=401-800
snapshots=500,600,700,800
output=$TRAINING_DIRECTORY"
    MINESTUDIO_BC_OUTPUT_DIRECTORY="$TRAINING_DIRECTORY" \
    MINESTUDIO_BC_VARIANT=upper_lora \
    MINESTUDIO_BC_MAX_STEPS=800 \
    MINESTUDIO_BC_VALIDATION_INTERVAL=50 \
    MINESTUDIO_BC_EARLY_STOPPING_PATIENCE=100 \
    MINESTUDIO_BC_SNAPSHOT_STEPS=500,600,700,800 \
    MINESTUDIO_BC_TELEGRAM=1 \
    python run/train_stone_recovery_bc.py 2>&1 | tee -a "$SUITE_DIRECTORY/training.log"
}

materialize_snapshots() {
    phase=materialize_snapshots
    write_state running "$phase" none lora800
    MINESTUDIO_BC_OUTPUT_DIRECTORY="$TRAINING_DIRECTORY" \
    MINESTUDIO_BC_MATERIALIZE_STEPS=500,600,700,800 \
    python run/materialize_stone_recovery_snapshots.py 2>&1 | tee -a "$SUITE_DIRECTORY/materialize.log"
    MINESTUDIO_CHECKPOINT="$TRAINING_DIRECTORY/upper_lora/snapshots/step_800_model" python - <<'PY'
import json
import os

import torch
from minestudio.models import SteveOnePolicy


checkpoint = os.environ["MINESTUDIO_CHECKPOINT"]
model = SteveOnePolicy.from_pretrained(checkpoint).to("cuda").eval()
condition = model.prepare_condition({"cond_scale": 6.0, "text": "mine stone and collect cobblestone, obtain 3 cobblestone"}, deterministic=False)
image = torch.zeros((1, 1, 128, 128, 3), dtype=torch.uint8, device="cuda")
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output, state = model({"image": image, "condition": condition}, None)
result = {
    "buttons_finite": bool(torch.isfinite(output["pi_logits"]["buttons"]).all()),
    "camera_finite": bool(torch.isfinite(output["pi_logits"]["camera"]).all()),
    "recurrent_state": state is not None,
}
print(json.dumps(result))
if not all(result.values()):
    raise RuntimeError(result)
PY
}

checkpoint_for() {
    local policy=$1
    if test "$policy" = vanilla; then
        echo "$PROJECT_DIRECTORY/checkpoints/steve_one_official"
    elif test "$policy" = lora400; then
        echo "$SOURCE_TRAINING_DIRECTORY/upper_lora/best_model"
    else
        echo "$TRAINING_DIRECTORY/upper_lora/snapshots/step_800_model"
    fi
}

run_evaluation() {
    split=$1
    policy=$2
    local episodes=$3
    local base_seed=$4
    local manifest_kind=$5
    local experiment_name=stage1b_${split}_${policy}_v1
    local checkpoint
    checkpoint=$(checkpoint_for "$policy")
    phase=evaluation
    write_state running "$phase" "$split" "$policy"
    notify "MineStudio confirmation evaluation started
split=$split
policy=$policy
episodes=$episodes
manifest=$manifest_kind"
    MINESTUDIO_STONE_CHECKPOINT="$checkpoint" \
    MINESTUDIO_STONE_EXPERIMENT="$experiment_name" \
    MINESTUDIO_STONE_EPISODES="$episodes" \
    MINESTUDIO_STONE_WORKERS="$WORKERS" \
    MINESTUDIO_STONE_BASE_SEED="$base_seed" \
    MINESTUDIO_STONE_MANIFEST_KIND="$manifest_kind" \
    MINESTUDIO_STONE_STAGE="Stage 1b LoRA confirmation" \
    python run/benchmark_stone_acquisition.py 2>&1 | tee -a "$SUITE_DIRECTORY/${split}_${policy}.log"
    local success_rate
    success_rate=$(jq -r '.overall.success_rate' "$PROJECT_DIRECTORY/output/stone_acquisition/$experiment_name/summary.json")
    notify "MineStudio confirmation evaluation completed
split=$split
policy=$policy
success_rate=$success_rate"
}

python -m pip freeze > "$SUITE_DIRECTORY/packages.txt"
nvidia-smi > "$SUITE_DIRECTORY/nvidia-smi.txt"
notify "MineStudio LoRA-800 confirmation suite started
general=3x1000
hazard_stress=3x600
suite=$SUITE_DIRECTORY"

prepare_training
train_lora800
materialize_snapshots

for policy in vanilla lora400 lora800; do
    run_evaluation general "$policy" "$GENERAL_EPISODES" "$GENERAL_SEED" general_natural
done

for policy in vanilla lora400 lora800; do
    run_evaluation stress "$policy" "$STRESS_EPISODES" "$STRESS_SEED" hazard_stress
done

phase=summarize
write_state running "$phase" none none
python run/summarize_stone_recovery_confirmation.py > "$SUITE_DIRECTORY/confirmation_report.stdout.json"
write_state completed none none none
notify "MineStudio LoRA-800 confirmation suite completed
report=$SUITE_DIRECTORY/confirmation_report.json"
