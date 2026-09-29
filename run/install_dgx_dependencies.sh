#!/usr/bin/env bash

set -euo pipefail
project_directory=/home/boni/projects/MineStudio
source /home/boni/ai/envs/dgx-dl/bin/activate
python -m pip install -c "$project_directory/constraints-dgx.txt" hydra-core==1.3.2 dm-tree x_transformers==0.27.1 gym==0.26.2 hydra_colorlog
python -m pip install --no-deps gym3==0.3.3
python -m pip install -c "$project_directory/constraints-dgx.txt" daemoniker Pyro4 xmltodict lmdb lightning albumentations pyrender==0.1.45 'pyglet<2' imgui pyopengl ray minecraft_data==3.20.0 'google-genai>=2.23,<3' 'python-dotenv>=1.2,<2'
python -m pip install --no-deps -e "$project_directory"
