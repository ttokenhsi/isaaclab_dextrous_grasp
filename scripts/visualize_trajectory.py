"""Headless visualisation of the recorded reference trajectories.

For one episode of a single env, this script:

* Lets the env reset normally so the robot lands in its pre-grasp init pose.
* Each control step, kinematically teleports the object (via
  ``write_root_pose_to_sim``) to ``env._traj_obj_t / _traj_obj_q`` so the
  bottle exactly traces the recorded MANO trajectory.
* Spawns 6 ``VisualizationMarkers`` spheres -- 1 per palm + 4 fingertip
  reference points + 1 large sphere overlaying the object target.
* Steps the env with a zero action (the robot's PD just holds it at the
  pre-grasp pose); ``gym.wrappers.RecordVideo`` captures an MP4.

This makes it easy to verify, frame by frame, that the recorded
imitate-phase trajectory (object + finger targets) actually rises and that
the per-env yaw / xy randomisation is correctly applied.

Usage::

    conda activate env_isaaclab
    python scripts/visualize_trajectory.py \\
        --trajectory ycb-006_mustard_bottle-20200709-subject-01-20200709_143211 \\
        --num_steps 75 --video_dir /tmp/traj_vis
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI + AppLauncher (must run before any other isaaclab import)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Visualise the reference trajectory.")
parser.add_argument("--task", default="Isaac-AllegroUR5-Relocate-v0")
parser.add_argument("--stage", type=int, default=0)
parser.add_argument(
    "--trajectory",
    type=str,
    default=None,
    help="Trajectory name override (without .npz).",
)
parser.add_argument("--num_steps", type=int, default=75)
parser.add_argument("--video_dir", type=str, default="/tmp/traj_vis")
parser.add_argument("--video_length", type=int, default=75)
parser.add_argument(
    "--name_prefix", type=str, default="trajectory", help="Output mp4 filename prefix."
)
parser.add_argument(
    "--cam_eye",
    type=float,
    nargs=3,
    default=[1.00, 0.95, 0.55],
    metavar=("X", "Y", "Z"),
    help=(
        "Camera eye position (env-local frame, meters). Default sits ``NE`` "
        "of the canonical object pose (0.35, 0.35) at ``55 cm`` above the "
        "table -- same family as ``train.py``'s recording cam, so the "
        "robot+object share the frame at a natural ~30° downward angle."
    ),
)
parser.add_argument(
    "--cam_lookat",
    type=float,
    nargs=3,
    default=[0.35, 0.35, 0.05],
    metavar=("X", "Y", "Z"),
    help=(
        "Camera target position (env-local frame, meters). Default ``z=0.05`` "
        "is intentionally low so flat objects (banana, plate, ...) sit in "
        "the centre of frame instead of being clipped against the table edge."
    ),
)
parser.add_argument(
    "--cam_resolution",
    type=int,
    nargs=2,
    default=[1280, 720],
    metavar=("W", "H"),
    help="Recorded video resolution.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless + cameras (needed for video).
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that depend on the IsaacSim app being launched.
# ---------------------------------------------------------------------------

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import isaaclab_dextrous_grasp  # noqa: F401  (registers gym IDs)
from isaaclab_dextrous_grasp.tasks.allegro_relocate.manager_env_cfg import (
    AllegroRelocateManagerEnvCfg,
)


def _make_marker_cfg() -> VisualizationMarkersCfg:
    """5 prototypes: palm + 4 fingertip reference targets.

    The object's reference pose is *not* drawn as a separate marker -- the
    actual YCB mesh is already teleported to that pose every step via
    ``write_root_pose_to_sim``, so the rendered object IS the target.

    A 6th cyan sphere used to overlay the object COM here, but ``PreviewSurface``
    opacity is unreliable in IsaacSim's MP4 render pass (the alpha gets
    flattened to 1.0), and a 3 cm sphere completely hid flat objects like
    the banana whose cross-section is also ~3 cm. The "halo above object"
    indicator below is the lightweight replacement.
    """

    return VisualizationMarkersCfg(
        prim_path="/Visuals/TrajTargets",
        markers={
            "palm": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0), opacity=1.0
                ),
            ),
            "thumb": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.15, 0.15), opacity=1.0
                ),
            ),
            "index": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 1.0, 0.15), opacity=1.0
                ),
            ),
            "middle": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.20, 0.40, 1.0), opacity=1.0
                ),
            ),
            "ring": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.20), opacity=1.0
                ),
            ),
            # Small floating cyan halo to mark "where the object should be"
            # without obscuring the actual object underneath. Sits 8 cm above
            # the object COM each frame.
            "object_halo": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0), opacity=1.0
                ),
            ),
        },
    )


def main() -> None:
    env_cfg = AllegroRelocateManagerEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.task.stage = args_cli.stage
    if args_cli.trajectory is not None:
        env_cfg.task.trajectory_name = args_cli.trajectory

    # We are kinematically driving the object every step; disable the
    # premature termination terms so a single episode plays through the
    # full ``imitate_steps`` window. Keep ``time_out`` so the env still
    # bounds itself.
    env_cfg.terminations.pregrasp_failure = None
    env_cfg.terminations.lost_contact = None
    env_cfg.terminations.object_too_far = None

    # Override the recording camera: env-local frame, close to the bottle.
    env_cfg.viewer.eye = tuple(args_cli.cam_eye)
    env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.resolution = tuple(args_cli.cam_resolution)

    os.makedirs(args_cli.video_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args_cli.video_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
        name_prefix=args_cli.name_prefix,
    )

    obs, _ = env.reset()
    inner = env.unwrapped

    markers = VisualizationMarkers(_make_marker_cfg())

    # Marker prototype order matches our cfg dict insertion order:
    #   0 palm, 1 thumb, 2 index, 3 middle, 4 ring, 5 object_halo
    proto_idx = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long, device=inner.device)
    # Vertical offset for the cyan "object halo" so it floats above the
    # actual object mesh instead of overlapping it (which used to hide
    # flat objects like the banana).
    OBJECT_HALO_Z_OFFSET = 0.08

    pregrasp_steps = int(inner._pregrasp_steps[0].item())
    imitate_steps = int(inner._imitate_steps[0].item())
    print(
        f"[INFO] Episode plan: pregrasp_steps={pregrasp_steps}, "
        f"imitate_steps={imitate_steps}, num_steps={args_cli.num_steps}"
    )

    env_origins = inner.scene.env_origins  # (E, 3) global offsets
    num_envs = env_cfg.scene.num_envs

    zero_action = torch.zeros((num_envs, 22), device=inner.device)
    zero_vel = torch.zeros((num_envs, 6), device=inner.device)

    for step in range(args_cli.num_steps):
        cs = int(inner.current_step[0].item())
        in_pregrasp = cs <= pregrasp_steps

        if in_pregrasp:
            # Pre-grasp marker = last frame of the pregrasp jpos trajectory
            jpos_local = inner._traj_pregrasp[:, -1]  # (E, 5, 3)  [palm, t, i, m, r]
            obj_t_local = inner._traj_obj_t[:, 0]
            obj_q_local = inner._traj_obj_q[:, 0]
        else:
            imitate_idx = min(cs - pregrasp_steps, inner._traj_jpos.shape[1] - 1)
            jpos_local = inner._traj_jpos[:, imitate_idx]
            obj_t_local = inner._traj_obj_t[:, imitate_idx]
            obj_q_local = inner._traj_obj_q[:, imitate_idx]

        # env-local → global world (markers and write_root_pose_to_sim expect global)
        jpos_global = jpos_local + env_origins.unsqueeze(1)            # (E, 5, 3)
        obj_pos_global = obj_t_local + env_origins                     # (E, 3)

        # Build (6, 3) marker translations for env 0:
        # - 5 fingertip / palm reference dots
        # - 1 cyan halo floating ``OBJECT_HALO_Z_OFFSET`` above the actual
        #   object mesh (the mesh itself is already teleported to
        #   ``obj_pos_global`` by ``write_root_pose_to_sim`` below).
        halo_pos = obj_pos_global[0:1].clone()
        halo_pos[:, 2] += OBJECT_HALO_Z_OFFSET
        marker_pos = torch.cat([jpos_global[0], halo_pos], dim=0)  # (6, 3)
        markers.visualize(translations=marker_pos, marker_indices=proto_idx)

        # Teleport the actual object to the target (kinematic playback).
        obj_pose_w = torch.cat([obj_pos_global, obj_q_local], dim=-1)  # (E, 7)
        inner.scene["object"].write_root_pose_to_sim(obj_pose_w)
        inner.scene["object"].write_root_velocity_to_sim(zero_vel)

        # Zero action: the robot just holds the pre-grasp pose under PD.
        obs, _, terminated, truncated, _ = env.step(zero_action)

        if step < 5 or step % 10 == 0:
            print(
                f"  step={step:3d}  cs={cs:3d}  phase={'pre' if in_pregrasp else 'imitate':7s}  "
                f"obj_z(env-local)={obj_t_local[0, 2].item():.4f}  "
                f"palm_marker_z(env-local)={jpos_local[0, 0, 2].item():.4f}"
            )

        if bool(truncated[0]):
            print(f"[INFO] truncated at step {step} (cs={cs})")
            break

    env.close()
    print(f"[INFO] Video saved under: {args_cli.video_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
