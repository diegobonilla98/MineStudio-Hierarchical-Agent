#!/usr/bin/env bash

set -euo pipefail
source /home/boni/projects/MineStudio/run/dgx_env.sh
cd /home/boni/projects/MineStudio
python run/test_pretrained_models.py
