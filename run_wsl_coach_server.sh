#!/usr/bin/env bash
set -e
export MINESTUDIO_DIR=/home/diego/.cache/minestudio
export HF_HOME=/home/diego/.cache/huggingface
export MAMBA_ROOT_PREFIX=/home/diego/.local/share/mamba
cd /mnt/d/HugeProjects/MineStudio
exec /home/diego/.local/bin/micromamba run -n minestudio python minestudio/tutorials/simulator/gemini_coach.py
