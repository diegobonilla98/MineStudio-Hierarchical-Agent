# MineStudio local setup

The complete simulator stack is installed in WSL2 because MineStudio's launcher explicitly rejects native Windows.

## Installed environment

- WSL distribution: `Ubuntu-24.04`
- Micromamba: `/home/diego/.local/bin/micromamba`
- Environment: `/home/diego/.local/share/mamba/envs/minestudio`
- Python: `3.11`
- Java: OpenJDK `8`
- PyTorch: `2.11.0+cu130`
- Torchvision: `0.26.0+cu130`
- Torchaudio: `2.11.0+cu130`
- Simulator assets: `/home/diego/.cache/minestudio`
- Hugging Face cache: `/home/diego/.cache/huggingface`

The Windows environment `C:\Users\diego\anaconda3\envs\python311` was not modified.

## Run the simulator smoke test

```powershell
& "D:\HugeProjects\MineStudio\run_wsl_smoke.ps1"
```

## Play and record demonstrations

```powershell
& "D:\HugeProjects\MineStudio\run_wsl_play.ps1"
```

In the MineStudio window, use `C` to capture or release the cursor, `Esc` for command mode, and `R` in command mode to start or stop recording. Recordings are written to `D:\HugeProjects\MineStudio\output`.

## Play with the Gemini task coach

```powershell
& "D:\HugeProjects\MineStudio\run_wsl_coach.ps1"
```

The launcher starts Minecraft and Gemini in WSL and opens a native Windows viewer using `C:\Users\diego\anaconda3\envs\python311\python.exe`. This bypasses the broken WSLg/pyglet presentation path while leaving the Windows PyTorch packages untouched. It reads `GEMINI_API_KEY` from `D:\HugeProjects\MineStudio\.env` and uses `gemini-3.8-flash`. Gemini reviews up to 12 sampled frames and issues one task at a time. Completion is checked from MineStudio state using inventory, event-count, or traveled-distance verifiers; visual model judgment cannot mark a task complete.

The first Minecraft launch can take around one minute. The legacy Gym notice, missing narrator library, Realms authorization, OpenAL audio, and OptiFine texture messages are non-fatal. The Windows `python311` environment is not used by this launcher.

The native gameplay window is titled `MineStudio Gemini Coach` and appears immediately with a loading status. The live frame appears after Minecraft finishes loading and Gemini creates the first task. Server output is written to `D:\HugeProjects\MineStudio\output\coach-server.stdout.log` and `coach-server.stderr.log`.

## Run another command in the environment

```powershell
$linuxCommand = "MAMBA_ROOT_PREFIX=/home/diego/.local/share/mamba /home/diego/.local/bin/micromamba run -n minestudio python -V"
wsl.exe -d Ubuntu-24.04 -- bash -lc $linuxCommand
```

## Recommended compute split

Use WSL2 on this PC for Minecraft simulation, interactive human demonstrations, and short inference checks. Keep the environment and policy loop on the same machine; do not send frame-by-frame actions over SSH.

Use `dgx-spark` for offline behavior cloning and RL optimization after trajectories are recorded and synchronized. The DGX host is ARM64, so the bundled MineStudio Minecraft runtime has not been validated there; treat it as a training host rather than the simulator host.
