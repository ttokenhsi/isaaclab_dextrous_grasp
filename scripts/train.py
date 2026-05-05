"""Train an RL agent on the AllegroUR5 YCB relocate task using rsl_rl PPO.

Usage::

    conda activate env_isaaclab
    python scripts/train.py --task Isaac-AllegroUR5-Relocate-v0 \
        --num_envs 64 --headless --max_iterations 5000

The script intentionally avoids Hydra so that the ``isaaclab_dextrous_grasp``
package is fully self-contained and easy to vendor.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

# IsaacLab AppLauncher MUST be created before any other isaaclab import.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO on AllegroUR5 relocate.")
parser.add_argument("--task", type=str, default="Isaac-AllegroUR5-Relocate-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--stage", type=int, default=0, help="Curriculum stage 0/1/2.")
parser.add_argument(
    "--trajectory",
    type=str,
    default=None,
    help="Trajectory name override (without .npz). If None, use the cfg default.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint to resume from.",
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos of env_0 during training."
)
parser.add_argument(
    "--video_length", type=int, default=200, help="Length (env steps) of each recorded video."
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Number of env steps between video recordings.",
)
parser.add_argument(
    "--video_cam_eye",
    type=float,
    nargs=3,
    default=[2.0, 2.0, 2.0],
    metavar=("X", "Y", "Z"),
    help=(
        "Recording camera eye, in env-local frame (m). Default keeps IsaacLab's "
        "(1,1,1) isometric direction but at ~3.5 m distance instead of the "
        "stock (7.5,7.5,7.5) which is ~13 m away."
    ),
)
parser.add_argument(
    "--video_cam_lookat",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.0],
    metavar=("X", "Y", "Z"),
    help="Recording camera lookat, in env-local frame (m). Default = env origin.",
)
parser.add_argument(
    "--video_resolution",
    type=int,
    nargs=2,
    default=[1280, 720],
    metavar=("W", "H"),
    help="Recorded video resolution.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Video recording requires cameras to be enabled in the AppLauncher.
if args_cli.video:
    args_cli.enable_cameras = True

# launch IsaacSim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that depend on the IsaacSim app being launched.
# ---------------------------------------------------------------------------

import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

# This import has the side-effect of registering the gym task IDs.
import isaaclab_dextrous_grasp  # noqa: F401
from isaaclab_dextrous_grasp.tasks.allegro_relocate.agents.rsl_rl_ppo_cfg import (
    AllegroUR5RelocatePPORunnerCfg,
)
from isaaclab_dextrous_grasp.tasks.allegro_relocate.manager_env_cfg import (
    AllegroRelocateManagerEnvCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg


def main() -> None:
    env_cfg = AllegroRelocateManagerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.task.stage = args_cli.stage
    if args_cli.trajectory is not None:
        env_cfg.task.trajectory_name = args_cli.trajectory
    env_cfg.seed = args_cli.seed

    # Override the recording camera to a close-up over env 0's tabletop so the
    # video focuses on the actual grasp instead of the wide simulation scene.
    if args_cli.video:
        env_cfg.viewer.eye = tuple(args_cli.video_cam_eye)
        env_cfg.viewer.lookat = tuple(args_cli.video_cam_lookat)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.resolution = tuple(args_cli.video_resolution)

    agent_cfg = AllegroUR5RelocatePPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "train")
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to: {video_dir}")
        print(
            f"[INFO]   length={args_cli.video_length} steps,"
            f" interval={args_cli.video_interval} steps"
        )
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args_cli.checkpoint is not None and os.path.isfile(args_cli.checkpoint):
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)

    start = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training finished in {time.time() - start:.1f}s")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
