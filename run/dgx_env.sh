#!/usr/bin/env bash

source /home/boni/ai/envs/dgx-dl/bin/activate
if test -f /home/boni/.config/minestudio/telegram.env; then
    source /home/boni/.config/minestudio/telegram.env
fi
export MINESTUDIO_DIR=/home/boni/.cache/minestudio
export HF_HOME=/home/boni/.cache/huggingface
export PYTHONPATH=/home/boni/ai/apps/TelegramBot:/home/boni/projects/MineStudio/third_party/ROCKET-2:/home/boni/projects/MineStudio${PYTHONPATH:+:$PYTHONPATH}
