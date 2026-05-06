"""Play a trained policy on the AllegroUR5 YCB relocate task.

Usage (basic, no video)::

    python scripts/play.py --task Isaac-AllegroUR5-Relocate-v0 \
        --num_envs 4 --checkpoint logs/.../model_5000.pt

Usage (record a short headless video on env 0)::

    python scripts/play.py --task Isaac-AllegroUR5-Relocate-v0 \
        --num_envs 1 --headless --video --num_steps 90 \
        --checkpoint logs/rsl_rl/allegro_ur5_relocate/2026-05-06_16-05-49/model_500.pt

The video lands next to the checkpoint at
``<ckpt_log_dir>/videos/play/rl-video-step-0.mp4`` by default.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a PPO policy.")
parser.add_argument("--task", type=str, default="Isaac-AllegroUR5-Relocate-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--trajectory", type=str, default=None)
parser.add_argument("--num_steps", type=int, default=400)

# Video recording (headless friendly)
parser.add_argument(
    "--video",
    action="store_true",
    help="If set, record a video of the rollout (uses env 0).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Number of env steps to record (defaults to --num_steps).",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help=(
        "Where to save the video. Defaults to "
        "``<dirname(checkpoint)>/videos/play/``."
    ),
)
parser.add_argument(
    "--video_cam_eye",
    type=float,
    nargs=3,
    default=[2.0, 2.0, 2.0],
    metavar=("X", "Y", "Z"),
    help=(
        "Recording camera eye, in env-local frame (m). Default keeps the "
        "(1,1,1) isometric direction at ~3.5 m distance."
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata
import torch
from rsl_rl.runners import OnPolicyRunner

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

    if args_cli.video:
        env_cfg.viewer.eye = tuple(args_cli.video_cam_eye)
        env_cfg.viewer.lookat = tuple(args_cli.video_cam_lookat)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.resolution = tuple(args_cli.video_resolution)

    agent_cfg = AllegroUR5RelocatePPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        if args_cli.video_dir is None:
            ckpt_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
            video_dir = os.path.join(ckpt_dir, "videos", "play")
        else:
            video_dir = os.path.abspath(args_cli.video_dir)
        os.makedirs(video_dir, exist_ok=True)
        video_length = args_cli.video_length or args_cli.num_steps
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to: {video_dir}")
        print(f"[INFO]   length={video_length} steps, resolution={tuple(args_cli.video_resolution)}")

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # IMPORTANT: explicitly reset *before* the first ``step``. Without this,
    # ``gym.wrappers.RecordVideo`` never sees a ``reset`` callback and its
    # ``step_trigger=lambda s: s == 0`` never fires, so the recording state
    # machine wedges and the first ``env.step`` blocks indefinitely (we
    # observed >3 min hangs at 100% CPU). ``RslRlVecEnvWrapper`` does not
    # reset on construction, so we have to do it here.
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    print(f"[INFO] env reset complete; running {args_cli.num_steps} steps...")

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
        actions = actions.clone()
        # ``RslRlVecEnvWrapper.step`` returns a 4-tuple (obs, rew, dones, extras).
        obs, _, _, _ = env.step(actions)
        if (step + 1) % 10 == 0:
            print(f"[INFO] step {step + 1}/{args_cli.num_steps}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
