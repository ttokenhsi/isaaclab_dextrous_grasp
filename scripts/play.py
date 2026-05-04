"""Play a trained policy on the AllegroUR5 YCB relocate task.

Usage::

    python scripts/play.py --task Isaac-AllegroUR5-Relocate-v0 \
        --num_envs 4 --checkpoint logs/.../model_5000.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a PPO policy.")
parser.add_argument("--task", type=str, default="Isaac-AllegroUR5-Relocate-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--trajectory", type=str, default=None)
parser.add_argument("--num_steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
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
import importlib.metadata as metadata


def main() -> None:
    env_cfg = AllegroRelocateManagerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.task.stage = args_cli.stage
    if args_cli.trajectory is not None:
        env_cfg.task.trajectory_name = args_cli.trajectory

    agent_cfg = AllegroUR5RelocatePPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    obs, _ = env.get_observations()
    for _ in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
