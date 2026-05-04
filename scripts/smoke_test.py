"""Smoke test: make the env, step it a few times, print obs / reward shapes.

Usage::

    python scripts/smoke_test.py --num_envs 4 --num_steps 20 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for AllegroUR5 relocate.")
parser.add_argument("--task", type=str, default="Isaac-AllegroUR5-Relocate-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=20)
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--trajectory", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_dextrous_grasp  # noqa: F401
from isaaclab_dextrous_grasp.tasks.allegro_relocate.manager_env_cfg import (
    AllegroRelocateManagerEnvCfg,
)


def main() -> None:
    env_cfg = AllegroRelocateManagerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.task.stage = args_cli.stage
    if args_cli.trajectory is not None:
        env_cfg.task.trajectory_name = args_cli.trajectory

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] action_space  : {env.action_space}")
    print(f"[INFO] observation_space: {env.observation_space}")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    print(f"[INFO] obs.shape = {obs.shape}  (expected (E, 393))")
    assert obs.shape[-1] == 393, f"expected 393 obs dims, got {obs.shape[-1]}"

    for step in range(args_cli.num_steps):
        actions = torch.zeros((args_cli.num_envs, 22), device=obs.device)
        obs_dict, reward, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        if step % 5 == 0:
            print(
                f"step={step:3d}  reward.mean={reward.float().mean():+.3f}  "
                f"terminated={int(terminated.sum())}  truncated={int(truncated.sum())}"
            )

    print("[INFO] Smoke test PASSED.")
    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
