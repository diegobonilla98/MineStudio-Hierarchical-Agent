#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY=/home/boni/projects/MineStudio
TRAINING_DIRECTORY=${MINESTUDIO_BC_OUTPUT_DIRECTORY:-$PROJECT_DIRECTORY/output/stone_recovery_bc/latest_run}
DATASET_DIRECTORY=$PROJECT_DIRECTORY/output/stone_recovery_bc/dataset_v1

cd "$PROJECT_DIRECTORY"
source run/dgx_env.sh
export MINESTUDIO_BC_OUTPUT_DIRECTORY="$TRAINING_DIRECTORY"
python run/select_stone_recovery_checkpoint.py

VARIANT=$(jq -r '.selected.variant' "$TRAINING_DIRECTORY/selection.json")
CHECKPOINT=$(jq -r '.selected.checkpoint_directory' "$TRAINING_DIRECTORY/selection.json")
EXPERIMENT_NAME="stage1_bc_${VARIANT}_v1"
EXPERIMENT_DIRECTORY=$PROJECT_DIRECTORY/output/stone_acquisition/$EXPERIMENT_NAME
mkdir -p "$EXPERIMENT_DIRECTORY"
cp "$DATASET_DIRECTORY/frozen_manifest.jsonl" "$EXPERIMENT_DIRECTORY/manifest.jsonl"

MINESTUDIO_STONE_CHECKPOINT="$CHECKPOINT" MINESTUDIO_STONE_EXPERIMENT="$EXPERIMENT_NAME" python run/benchmark_stone_acquisition.py
