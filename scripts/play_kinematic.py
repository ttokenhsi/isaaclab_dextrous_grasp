"""Kinematic replay of a recorded ViViDex rollout in IsaacLab.

Loads a ``.npz`` produced by ``vividex_sapien/tools/record_replay_data.py``
which contains, for every step of the rollout:

* ``robot_qpos`` -- (T, 22) joint positions in SAPIEN URDF declaration order
  ``[arm_0..arm_5, joint_00, joint_01, ..., joint_15]``.
* ``object_pos`` -- (T, 3) object position in the SAPIEN world frame.
* ``object_quat`` -- (T, 4) object quaternion in SAPIEN ``wxyz`` order.

For each frame the script **teleports** the robot's joints and the object
pose directly into PhysX via ``write_joint_position_to_sim`` /
``set_joint_position_target`` / ``write_root_pose_to_sim``. This bypasses
the policy and the IK action manager entirely -- the simulator is being
used as a pure forward-kinematics renderer. The point is to verify that
the per-step state recorded in SAPIEN, when re-rendered through
IsaacLab's USD assets / cameras, *looks the same as the SAPIEN video* --
i.e. the policy + physics gap is excluded, and only mesh / link / asset
mismatches remain visible.

Usage::

    conda activate env_isaaclab
    python scripts/play_kinematic.py \\
        --task Isaac-AllegroUR5-Relocate-v0 \\
        --replay_data /tmp/replay/actions_ycb-006_mustard_bottle-...stage0.npz \\
        --video_dir logs/kinematic_replay \\
        --num_steps 60
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI + AppLauncher (must run before any other isaaclab import)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Kinematic teleport replay of a ViViDex rollout.")
parser.add_argument("--task", default="Isaac-AllegroUR5-Relocate-v0")
parser.add_argument(
    "--replay_data",
    type=str,
    required=True,
    help="Path to the .npz produced by record_replay_data.py (must contain robot_qpos).",
)
parser.add_argument("--stage", type=int, default=0)
parser.add_argument(
    "--trajectory",
    type=str,
    default=None,
    help="Trajectory name override (without .npz). If None, uses the cfg default.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=None,
    help="How many recorded frames to replay. Defaults to the full rollout length.",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record an mp4 of the kinematic replay.",
)
parser.add_argument("--video_dir", type=str, default="logs/kinematic_replay")
parser.add_argument(
    "--name_prefix", type=str, default="kinematic", help="Output mp4 filename prefix."
)
parser.add_argument(
    "--cam_eye",
    type=float,
    nargs=3,
    default=[1.05, 0.95, 0.55],
    metavar=("X", "Y", "Z"),
    help="Camera eye, env-local frame (m). Default matches record_replay_data.py.",
)
parser.add_argument(
    "--cam_lookat",
    type=float,
    nargs=3,
    default=[0.40, 0.40, 0.18],
    metavar=("X", "Y", "Z"),
    help="Camera lookat, env-local frame (m). Default matches record_replay_data.py.",
)
parser.add_argument(
    "--cam_resolution",
    type=int,
    nargs=2,
    default=[640, 480],
    metavar=("W", "H"),
    help="Recorded video resolution.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless when recording -- the Kit GUI fights with offscreen render.
args_cli.headless = True
if args_cli.video:
    # Replicator must be initialised at AppLauncher boot time, otherwise the
    # first env.render() call kicks off a 60-90 s lazy-init. See the same
    # comment in play_vividex.py / play_replay.py.
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that depend on the IsaacSim app being launched.
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_dextrous_grasp  # noqa: F401  (registers gym IDs)  E402
from isaaclab_dextrous_grasp.tasks.allegro_relocate.manager_env_cfg import (  # noqa: E402
    AllegroRelocateManagerEnvCfg,
)


def _build_inverse_qpos_idx(sapien_idx: torch.Tensor) -> torch.Tensor:
    """Invert the URDF->lab joint permutation.

    ``sapien_idx[k] == lab_joint_id`` tells us where the k-th URDF joint
    sits inside IsaacLab's articulation tensor. To copy a SAPIEN qpos
    vector ``q_sapien`` (URDF order) *into* the lab tensor we need the
    inverse: ``q_lab[inv[i]] = q_sapien[i]``, i.e. ``q_lab = q_sapien[inv]``
    with ``inv = argsort(sapien_idx)``.
    """

    inv = torch.argsort(sapien_idx)
    return inv


def main() -> None:
    replay_path = Path(args_cli.replay_data)
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay npz not found: {replay_path}")
    data = np.load(replay_path)
    required = ("robot_qpos", "object_pos", "object_quat")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(
            f"Replay npz is missing {missing}. Re-record with the updated "
            "record_replay_data.py (it now dumps robot_qpos / robot_qvel)."
        )

    robot_qpos_all = data["robot_qpos"].astype(np.float32)   # (T, 22) URDF order
    object_pos_all = data["object_pos"].astype(np.float32)   # (T, 3)
    object_quat_all = data["object_quat"].astype(np.float32) # (T, 4) wxyz
    T = robot_qpos_all.shape[0]
    num_steps = T if args_cli.num_steps is None else min(args_cli.num_steps, T)
    print(f"[INFO] loaded {T} frames from {replay_path.name}, replaying {num_steps}")

    # -------- build env cfg --------
    env_cfg = AllegroRelocateManagerEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.task.stage = args_cli.stage
    if args_cli.trajectory is not None:
        env_cfg.task.trajectory_name = args_cli.trajectory

    # Disable every early-termination term -- the kinematic teleport will
    # frequently cause "object_too_far" / "lost_contact" triggers that have
    # no meaning here (we are bypassing policy + physics).
    env_cfg.terminations.pregrasp_failure = None
    env_cfg.terminations.lost_contact = None
    env_cfg.terminations.object_too_far = None

    # Camera in env-local frame, framing the bottle + hand.
    env_cfg.viewer.eye = tuple(args_cli.cam_eye)
    env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.resolution = tuple(args_cli.cam_resolution)

    os.makedirs(args_cli.video_dir, exist_ok=True)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=num_steps,
            disable_logger=True,
            name_prefix=args_cli.name_prefix,
        )

    obs, _ = env.reset()
    inner = env.unwrapped
    device = inner.device

    robot = inner.scene["robot"]
    obj = inner.scene["object"]

    sapien_idx = inner._qpos_sapien_idx              # (22,) URDF k -> lab joint id
    inv_idx = _build_inverse_qpos_idx(sapien_idx)    # (22,) lab joint id k -> URDF idx
    print(f"[INFO] sapien_idx={sapien_idx.tolist()}")
    print(f"[INFO] inv_idx   ={inv_idx.tolist()}")

    env_origins = inner.scene.env_origins             # (E, 3) global offsets
    num_envs = env_cfg.scene.num_envs
    zero_vel_root = torch.zeros((num_envs, 6), device=device)
    zero_qvel = torch.zeros((num_envs, 22), device=device)
    zero_action = torch.zeros((num_envs, 22), device=device)

    qpos_buf = torch.zeros((num_envs, 22), device=device)
    obj_pose_buf = torch.zeros((num_envs, 7), device=device)

    for step in range(num_steps):
        # ---- Build this frame's lab-order qpos from the SAPIEN-order log ----
        q_sapien = torch.from_numpy(robot_qpos_all[step]).to(device=device, dtype=torch.float32)
        # q_lab[sapien_idx[k]] = q_sapien[k]  <=>  q_lab = q_sapien[inv_idx]
        # We need q_lab indexed by lab joint id k, so use the scatter form:
        q_lab = torch.empty(22, device=device, dtype=torch.float32)
        q_lab[sapien_idx] = q_sapien
        qpos_buf[0] = q_lab

        # ---- Object pose: SAPIEN frame == env-local; add env origin to go global. ----
        op = torch.from_numpy(object_pos_all[step]).to(device=device, dtype=torch.float32)
        oq = torch.from_numpy(object_quat_all[step]).to(device=device, dtype=torch.float32)
        obj_pose_buf[0, :3] = op + env_origins[0]
        obj_pose_buf[0, 3:] = oq

        # ---- Teleport joints + object, then run a single sim step so the ----
        # ---- USD render reflects the new pose. ``set_joint_position_target`` ----
        # ---- prevents the PD from snapping back between frames.            ----
        robot.write_joint_state_to_sim(qpos_buf, zero_qvel)
        robot.set_joint_position_target(qpos_buf)
        obj.write_root_pose_to_sim(obj_pose_buf)
        obj.write_root_velocity_to_sim(zero_vel_root)

        obs, _, terminated, truncated, _ = env.step(zero_action)

        if step < 5 or step % 10 == 0:
            print(
                f"  step={step:3d}  obj_z(env-local)={object_pos_all[step, 2]:.4f}  "
                f"q_arm0={q_sapien[0]:.3f}  q_index_root={q_sapien[6]:.3f}"
            )

        if bool(truncated[0]):
            print(f"[INFO] truncated at step {step}")
            break

    env.close()
    print(f"[INFO] Kinematic replay finished. Video under: {args_cli.video_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
