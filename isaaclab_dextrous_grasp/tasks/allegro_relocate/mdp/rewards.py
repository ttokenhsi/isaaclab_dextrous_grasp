"""Reward functions reproducing ``vividex.AllegroRelocateRLEnv.get_reward``.

Reward structure (per env step, computed pre-/post-pregrasp)::

    if current_step <= pregrasp_steps:
        r_pregrasp = 10 * exp(-10 * mean ||fingertip - pregrasp_target||)
    else:
        r_contact         = 0.5 * num_finger_contacts
        r_object_track    = 10 * exp(-50 * (||obj_pos - tgt|| + 0.1 * obj_rot_err))
        r_fingertip_track = 4  * exp(-10 * mean ||fingertip - ref||)
        r_lift_bonus      = 2.5 if (obj_z - init_object_height > 0.02) else 0

    common penalty:
        r_ctrl = -1e3 * cartesian_error**2
        r_act  = -0.01 * sum(clip(qvel, -1, 1)**2)

    total = (r_pregrasp + r_contact + r_object_track + r_fingertip_track
             + r_lift_bonus + r_ctrl + r_act) / 10

The ``/10`` factor is folded into the ``weight`` of each :class:`RewTerm`
in :mod:`manager_env_cfg`. Each function below returns the *per-component*
reward in ViViDex's *un-divided* magnitude; the manager multiplies by the
configured ``weight`` (which is ``1/10`` times the ViViDex magnitude).

Wait: that approach is suboptimal because weights are different per term.
Instead each function returns the value *already divided by 10* and the
``weight`` in the cfg is set to ``1``. This keeps the cfg readable and the
sum equal to ViViDex's per-step total reward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..manager_env import AllegroRelocateManagerEnv


# ---------------------------------------------------------------------------
# Intermediate quantity cache
# ---------------------------------------------------------------------------


def _ensure_intermediates(env: "AllegroRelocateManagerEnv") -> dict:
    """Compute (and cache) per-step quantities shared by multiple rewards.

    The cache is populated once per env step and invalidated by
    :py:meth:`AllegroRelocateManagerEnv.step`. Subsequent reward terms in
    the same step read directly from ``env._reward_cache``.
    """

    if getattr(env, "_reward_cache", None) is not None:
        return env._reward_cache

    robot = env.scene["robot"]
    obj = env.scene["object"]

    # ---- positions -------------------------------------------------------
    finger_pos = robot.data.body_pos_w[:, env._finger_body_idx]  # (E, 4, 3)
    obj_pos = obj.data.root_pos_w                                # (E, 3)
    obj_quat = obj.data.root_quat_w                              # (E, 4)

    # ---- index trajectory references ------------------------------------
    pregrasp_steps = env._pregrasp_steps  # (E,)
    current_step = env.current_step       # (E,)
    in_pregrasp = current_step <= pregrasp_steps  # (E,) bool

    env_idx = torch.arange(env.num_envs, device=env.device)
    # Pregrasp target is the LAST entry of robot_pregrasp_jpos[:, 1:] = the 4 fingertips
    pre_target = env._traj_pregrasp[env_idx, -1, 1:]  # (E, 4, 3)
    pre_err = torch.linalg.norm(finger_pos - pre_target, dim=-1).mean(dim=-1)  # (E,)

    # Imitate target: jpos[current_step - pregrasp_steps], skipping the palm (idx 0).
    imitate_idx = torch.clamp(current_step - pregrasp_steps, min=0, max=env._traj_jpos.shape[1] - 1)
    ref_jpos = env._traj_jpos[env_idx, imitate_idx, 1:]   # (E, 4, 3)
    fingertip_err = torch.linalg.norm(finger_pos - ref_jpos, dim=-1).mean(dim=-1)  # (E,)

    ref_obj_pos = env._traj_obj_t[env_idx, imitate_idx]   # (E, 3)
    ref_obj_quat = env._traj_obj_q[env_idx, imitate_idx]  # (E, 4)
    obj_com_err = torch.linalg.norm(obj_pos - ref_obj_pos, dim=-1)
    # Rotation distance via quaternion: angle = 2 * acos(|<q1, q2>|) ∈ [0, pi]
    dot = torch.abs((obj_quat * ref_obj_quat).sum(dim=-1)).clamp(max=1.0)
    obj_rot_err = (2.0 * torch.acos(dot)) / torch.pi  # normalized to [0, 1]

    # ---- contact buckets via ContactSensor ------------------------------
    # Each sensor has 1 body + 1 filter; force_matrix_w shape (N, 1, 1, 3).
    # We threshold ||F|| > 1e-2/dt → contact bool (matches vividex impulse threshold).
    sensors = [
        env.scene.sensors["palm_contact"],
        env.scene.sensors["thumb_contact"],
        env.scene.sensors["index_contact"],
        env.scene.sensors["middle_contact"],
        env.scene.sensors["ring_contact"],
    ]
    forces = []
    for s in sensors:
        fm = s.data.force_matrix_w  # (E, 1, 1, 3) or None
        if fm is None:
            forces.append(torch.zeros(env.num_envs, device=env.device))
            continue
        # squeeze → (E, 3) → norm
        forces.append(torch.linalg.norm(fm.view(env.num_envs, 3), dim=-1))
    force_stack = torch.stack(forces, dim=-1)  # (E, 5)
    contact_bool = (force_stack > 1.0).float()  # (E, 5) [palm, thumb, idx, mid, ring]
    num_finger_contacts = contact_bool[:, 1:].sum(dim=-1)  # 4 fingers
    has_contact = (contact_bool.sum(dim=-1) >= 1).float()  # palm or any finger

    # ---- object lift -----------------------------------------------------
    init_z = env._init_object_height  # (E,) scalar per env (same value if a single obj)
    obj_lift = torch.clamp(obj_pos[:, 2] - init_z, min=0.0)

    # ---- store hand_jpos_err & friends for terminations ----------------
    cache = {
        "in_pregrasp": in_pregrasp,
        "pre_err": pre_err,
        "fingertip_err": fingertip_err,
        "obj_com_err": obj_com_err,
        "obj_rot_err": obj_rot_err,
        "num_finger_contacts": num_finger_contacts,
        "has_contact": has_contact,
        "contact_bool": contact_bool,
        "obj_lift": obj_lift,
    }
    env._reward_cache = cache
    return cache


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def pregrasp_reward(env: "AllegroRelocateManagerEnv", err_scale: float = 10.0) -> torch.Tensor:
    cache = _ensure_intermediates(env)
    r = 10.0 * torch.exp(-err_scale * cache["pre_err"])  # (E,)
    return torch.where(cache["in_pregrasp"], r, torch.zeros_like(r)) / 10.0


def contact_reward(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    cache = _ensure_intermediates(env)
    r = 0.5 * cache["num_finger_contacts"]
    return torch.where(cache["in_pregrasp"], torch.zeros_like(r), r) / 10.0


def object_track_reward(
    env: "AllegroRelocateManagerEnv", err_scale: float = 50.0, rot_weight: float = 0.1
) -> torch.Tensor:
    cache = _ensure_intermediates(env)
    r = 10.0 * torch.exp(-err_scale * (cache["obj_com_err"] + rot_weight * cache["obj_rot_err"]))
    return torch.where(cache["in_pregrasp"], torch.zeros_like(r), r) / 10.0


def fingertip_track_reward(
    env: "AllegroRelocateManagerEnv", err_scale: float = 10.0
) -> torch.Tensor:
    cache = _ensure_intermediates(env)
    r = 4.0 * torch.exp(-err_scale * cache["fingertip_err"])
    return torch.where(cache["in_pregrasp"], torch.zeros_like(r), r) / 10.0


def lift_bonus_reward(
    env: "AllegroRelocateManagerEnv", thresh: float = 0.02
) -> torch.Tensor:
    cache = _ensure_intermediates(env)
    bonus = torch.where(cache["obj_lift"] > thresh, torch.full_like(cache["obj_lift"], 2.5), torch.zeros_like(cache["obj_lift"]))
    return torch.where(cache["in_pregrasp"], torch.zeros_like(bonus), bonus) / 10.0


def controller_penalty(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    """``-1e3 * cartesian_error**2 / 10``."""

    err = getattr(env, "_cartesian_error", None)
    if err is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (-1e3 * err.pow(2)) / 10.0


def action_penalty(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    """``-0.01 * sum(clip(qvel, -1, 1)**2) / 10``."""

    qvel = env.scene["robot"].data.joint_vel.clamp(min=-1.0, max=1.0)
    pen = -0.01 * (qvel * qvel).sum(dim=-1)
    return pen / 10.0
