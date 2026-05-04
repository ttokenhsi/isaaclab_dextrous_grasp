"""Observation functions reproducing the ViViDex 393-d oracle state.

Layout::

    robot_state   : 330  -- qpos(22) + qvel(22) + 14 link [pos+quat+linvel+angvel] = 22+22+14*13
    object_state  :  13  -- pos(3) + quat(4) + linvel(3) + angvel(3)
    goal_state    :  42  -- 3 future-frame [orn(4)+trans(3)] = 21 +
                            palm-obj diff(3) + 4 fingertip-obj diff(12) +
                            palm-target diff(3) + obj-target diff(3)
    time_state    :   8  -- sin/cos(t * [1, 4, 6, 8])

All four functions take ``env: AllegroRelocateManagerEnv`` and return a
``(num_envs, D)`` float32 tensor.

The env is expected to provide:

* ``env._joint_link_body_idx`` -- (14,) tensor of body indices for the 14
  joint links that contribute to ``robot_state``.
* ``env._palm_body_idx`` -- int, palm body index.
* ``env._finger_body_idx`` -- (4,) tensor of body indices for the 4 finger
  parents.
* ``env._object_root_pos_w`` / ``env._object_root_quat_w`` /
  ``env._object_root_lin_vel_w`` / ``env._object_root_ang_vel_w`` -- accessed
  via ``env.scene["object"].data``.
* ``env._traj_obj_t`` / ``env._traj_obj_q`` / ``env._traj_jpos`` --
  per-env trajectory buffers (E, T, ...).
* ``env._traj_target_pos`` -- (E, 3) per-env final target position (last frame
  of object_translation).
* ``env._traj_step``, ``env._pregrasp_steps``, ``env._imitate_steps``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..manager_env import AllegroRelocateManagerEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gather_links_blockwise(env: "AllegroRelocateManagerEnv", body_idx: torch.Tensor) -> torch.Tensor:
    """Return ``[all_pos | all_quat | all_lin_vel | all_ang_vel]`` flattened.

    This matches ViViDex's ``get_robot_state`` ordering exactly:
    ``concatenate([pos.reshape(-1), quat.reshape(-1), lin_vel.reshape(-1), ang_vel.reshape(-1)])``.
    Output shape is ``(E, K * 13)`` for ``K`` bodies.
    """

    robot = env.scene["robot"]
    pos = robot.data.body_pos_w[:, body_idx]            # (E, K, 3)
    quat = robot.data.body_quat_w[:, body_idx]           # (E, K, 4)
    lin_vel = robot.data.body_lin_vel_w[:, body_idx]    # (E, K, 3)
    ang_vel = robot.data.body_ang_vel_w[:, body_idx]    # (E, K, 3)
    e = robot.data.body_pos_w.shape[0]
    return torch.cat(
        [
            pos.reshape(e, -1),
            quat.reshape(e, -1),
            lin_vel.reshape(e, -1),
            ang_vel.reshape(e, -1),
        ],
        dim=-1,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Hamilton-conjugate (assuming unit quaternions)."""

    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


# ---------------------------------------------------------------------------
# ObsTerm functions
# ---------------------------------------------------------------------------


def robot_state(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    """qpos(22) + qvel(22) + 14 link [pos(3)+quat(4)+linvel(3)+angvel(3)] = 330."""

    robot = env.scene["robot"]
    qpos = robot.data.joint_pos                          # (E, 22)
    qvel = robot.data.joint_vel                          # (E, 22)
    links = _gather_links_blockwise(env, env._joint_link_body_idx)  # (E, 22*13)
    return torch.cat([qpos, qvel, links], dim=-1)


def object_state(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    """pos(3) + quat(4) + lin_vel(3) + ang_vel(3) = 13."""

    obj = env.scene["object"]
    pos = obj.data.root_pos_w
    quat = obj.data.root_quat_w
    lv = obj.data.root_lin_vel_w
    av = obj.data.root_ang_vel_w
    return torch.cat([pos, quat, lv, av], dim=-1)


def goal_state(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    """3 future-frame [orn(4)+trans(3)] + palm-obj diff(3) + 4 fingertip-obj
    diff(12) + palm-target diff(3) + obj-target diff(3) = 42."""

    robot = env.scene["robot"]
    obj = env.scene["object"]

    # ----- 3 future-frame goals (i ∈ {1, 5, 10}) -------------------------
    futures = []
    pregrasp_steps = env._pregrasp_steps
    imitate_steps = env._imitate_steps
    traj_step = env._traj_step
    for i in (1, 5, 10):
        future_step = traj_step + i  # (E,)
        # If still in pregrasp, use frame 0 of the imitate segment; else
        # min(future_step, imitate_steps - 1) - pregrasp_steps.
        in_pregrasp = future_step <= pregrasp_steps
        clipped = torch.clamp(future_step, max=imitate_steps - 1) - pregrasp_steps
        clipped = torch.where(in_pregrasp, torch.zeros_like(clipped), clipped)
        clipped = torch.clamp(clipped, min=0)

        # Index per-env trajectory tensors
        env_idx = torch.arange(env.num_envs, device=env.device)
        orn = env._traj_obj_q[env_idx, clipped]   # (E, 4)
        trans = env._traj_obj_t[env_idx, clipped]  # (E, 3)
        futures.append(orn)
        futures.append(trans)
    traj_goals = torch.cat(futures, dim=-1)  # (E, 21)

    # ----- palm-obj diff -------------------------------------------------
    palm_pos = robot.data.body_pos_w[:, env._palm_body_idx]
    obj_pos = obj.data.root_pos_w
    hand_obj_diff = palm_pos - obj_pos  # (E, 3)

    # ----- 4 fingertip-obj diff -----------------------------------------
    finger_pos = robot.data.body_pos_w[:, env._finger_body_idx]  # (E, 4, 3)
    hand_obj_dense_diff = finger_pos - obj_pos.unsqueeze(1)  # (E, 4, 3)

    # ----- palm-target & obj-target diff --------------------------------
    target_pos = env._traj_target_pos  # (E, 3)
    hand_tgt_diff = palm_pos - target_pos
    obj_tgt_diff = obj_pos - target_pos

    return torch.cat(
        [
            traj_goals,
            hand_obj_diff,
            hand_obj_dense_diff.reshape(env.num_envs, -1),
            hand_tgt_diff,
            obj_tgt_diff,
        ],
        dim=-1,
    )


def time_state(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    """``sin/cos(t * [1, 4, 6, 8])`` with ``t = traj_step / imitate_steps``."""

    t = env._traj_step.float() / env._imitate_steps.float().clamp(min=1.0)
    freqs = torch.tensor([1.0, 4.0, 6.0, 8.0], device=env.device)
    arg = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (E, 4)
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
