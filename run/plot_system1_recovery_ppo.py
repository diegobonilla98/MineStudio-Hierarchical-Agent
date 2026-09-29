import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


RUN_DIRECTORY = Path(os.environ["MINESTUDIO_PPO_OUTPUT"])
TASKS = (
    "RECOVER_CAMERA",
    "AVOID_DIGGING_TRAP",
    "REACQUIRE_STONE",
    "CLIMB_SHORE",
    "ESCAPE_HOLE",
    "AVOID_WATER",
)


def main() -> None:
    records = [json.loads(line) for line in (RUN_DIRECTORY / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    training = [record for record in records if record["split"] == "train"]
    development = [record for record in records if record["split"] == "development"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot([record["iteration"] for record in training], [record["success_rate"] for record in training], label="rollout success")
    axes[0, 0].plot([record["iteration"] for record in training], [record["mean_episode_return"] for record in training], label="mean return")
    axes[0, 0].set_title("Training rollouts")
    axes[0, 0].legend()
    axes[0, 1].plot([record["iteration"] for record in training], [record["ppo"]["reference_kl"] for record in training], label="reference KL")
    axes[0, 1].plot([record["iteration"] for record in training], [record["ppo"]["approximate_kl"] for record in training], label="PPO approximate KL")
    axes[0, 1].set_title("Policy drift")
    axes[0, 1].legend()
    axes[1, 0].plot([record["iteration"] for record in development], [record["weak_mean"] for record in development], marker="o", label="weak-task mean")
    axes[1, 0].plot([record["iteration"] for record in development], [record["normal_stone"]["success_rate"] for record in development], marker="o", label="normal stone")
    axes[1, 0].set_title("Behavioral checkpoint selection")
    axes[1, 0].legend()
    for task in TASKS:
        axes[1, 1].plot([record["iteration"] for record in development], [record["micro"][task]["success_rate"] for record in development], marker="o", label=task.lower())
    axes[1, 1].set_title("Development micro-benchmarks")
    axes[1, 1].legend(fontsize=7, ncol=2)
    for axis in axes.flat:
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.25)
    figure.savefig(RUN_DIRECTORY / "training_curves.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
