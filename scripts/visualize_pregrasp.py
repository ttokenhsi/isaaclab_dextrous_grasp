"""Visualise the pregrasp / imitate / current keypoints in the env.

Spawns 3 sets of coloured spheres per env:

* **red**   -- pregrasp target (palm + 4 fingertips, the *last* frame of
  ``robot_pregrasp_jpos``, which is what ``pregrasp_reward`` matches).
* **blue**  -- imitate-frame-0 reference (the *first* frame of ``robot_jpos``;
  this is where the policy must be by ``current_step == pregrasp_steps``).
* **green** -- current robot keypoints (palm + 4 finger parent links read
  from ``body_pos_w``). These should converge onto the red markers during
  pregrasp and onto the blue markers during imitate if the policy is
  working.

Saves an MP4 of env_0's third-person view to ``logs/pregrasp_viz/``.

Usage::

    python scripts/visualize_pregrasp.py --num_envs 4 --num_steps 60
    # by default headless + video; pass --no_headless to drive the GUI.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualise pregrasp / imitate keypoints.")
parser.add_argument("--task", type=str, default="Isaac-AllegroUR5-Relocate-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=60)
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--video_length", type=int, default=60)
parser.add_argument(
    "--out_dir",
    type=str,
    default="logs/pregrasp_viz",
    help="Where to dump the recorded video (relative to package root).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Default: headless + camera enabled (so we can record video).
if not getattr(args_cli, "headless", False) and not getattr(args_cli, "livestream", 0):
    args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import isaaclab_dextrous_grasp  # noqa: F401  -- registers the gym ID
from isaaclab_dextrous_grasp.tasks.allegro_relocate.manager_env_cfg import (
    AllegroRelocateManagerEnvCfg,
)


def _sphere_markers(prim_path: str, color: tuple[float, float, float], radius: float) -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "s": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def main() -> None:
    cfg = AllegroRelocateManagerEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.task.stage = args_cli.stage

    out_dir = os.path.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=out_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        name_prefix="pregrasp_viz",
        disable_logger=True,
    )

    env.reset()
    unw = env.unwrapped

    # Allocate three marker instancers. Each holds num_envs * 5 spheres.
    n = args_cli.num_envs
    pregrasp_markers = _sphere_markers(
        "/World/Visuals/PregraspTarget", color=(1.0, 0.15, 0.15), radius=0.012
    )
    imitate0_markers = _sphere_markers(
        "/World/Visuals/Imitate0Ref", color=(0.15, 0.35, 1.0), radius=0.012
    )
    current_markers = _sphere_markers(
        "/World/Visuals/CurrentKeypoints", color=(0.20, 0.95, 0.30), radius=0.010
    )

    env_origins = unw.scene.env_origins  # (E, 3)

    def _update_markers() -> None:
        # Pregrasp target = traj_pregrasp[:, -1, :, :] (palm + 4 fingertips)
        # Stored env-local; convert to world by adding env_origins.
        pre = unw._traj_pregrasp[:, -1]  # (E, 5, 3) env-local
        pre_w = pre + env_origins.unsqueeze(1)  # (E, 5, 3)
        pre_flat = pre_w.reshape(-1, 3)
        pregrasp_markers.visualize(translations=pre_flat)

        # Imitate frame 0 reference = traj_jpos[:, 0]
        ref0 = unw._traj_jpos[:, 0]  # (E, 5, 3) env-local
        ref0_w = ref0 + env_origins.unsqueeze(1)
        imitate0_markers.visualize(translations=ref0_w.reshape(-1, 3))

        # Current keypoints: palm + 4 fingers (in world frame already)
        robot = unw.scene["robot"]
        palm = robot.data.body_pos_w[:, unw._palm_body_idx].unsqueeze(1)  # (E, 1, 3)
        fingers = robot.data.body_pos_w[:, unw._finger_body_idx]           # (E, 4, 3)
        cur = torch.cat([palm, fingers], dim=1)                            # (E, 5, 3)
        current_markers.visualize(translations=cur.reshape(-1, 3))

    _update_markers()
    print(
        "[INFO] Visualising pregrasp keypoints:\n"
        "  red    = pregrasp target (palm + 4 fingertips, last pregrasp frame)\n"
        "  blue   = imitate-frame-0 reference\n"
        "  green  = current robot keypoints (palm + 4 finger parents)"
    )
    print(f"[INFO] num_envs={n}, stage={args_cli.stage}, num_steps={args_cli.num_steps}")
    print(f"[INFO] Recording to: {out_dir}/pregrasp_viz-step-0.mp4")

    actions = torch.zeros((n, 22), device=unw.device)
    for step in range(args_cli.num_steps):
        env.step(actions)
        _update_markers()
        if step % 10 == 0:
            # Print a snapshot of env-0's pregrasp & current palm in env-local frame
            e0_pre = unw._traj_pregrasp[0, -1, 0].cpu().numpy()
            e0_palm = (unw.scene["robot"].data.body_pos_w[0, unw._palm_body_idx]
                       - env_origins[0]).cpu().numpy()
            print(
                f"step={step:3d}  env0 palm@local={e0_palm}  "
                f"pregrasp_palm_target@local={e0_pre}"
            )

    print("[INFO] Done.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
