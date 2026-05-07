"""Reset event: re-randomise per-env trajectory + reset robot / object state.

Mirrors :py:meth:`AllegroRelocateRLEnv.reset` but vectorised across env_ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import trajectory as trajectory_module

if TYPE_CHECKING:
    from ..manager_env import AllegroRelocateManagerEnv


def reset_trajectory_state(
    env: "AllegroRelocateManagerEnv",
    env_ids: torch.Tensor,
    stage: int = 0,  # noqa: ARG001 -- kept for backward-compat, ignored
) -> None:
    """Re-sample (x, y, theta_z) per env and reset all per-env trajectory state.

    Steps (matches ViViDex's reset):

    0. *(curriculum bookkeeping)* push the just-finished episode's
       ``_pregrasp_success`` value (one bool per env in ``env_ids``) into
       the rolling success-rate buffer so the curriculum manager can decide
       when to promote the stage.
    1. Sample new (x, y, theta_z) for ``env_ids`` and overwrite per-env
       trajectory buffers via :func:`trajectory.apply_stage_randomisation`.
       The ``stage`` is read from ``env.cfg.task.stage`` at runtime so the
       curriculum can mutate it between resets.
    2. Recompute per-env final goal position (``traj_target_pos``).
    3. Reset state counters: ``current_step / traj_step / pregrasp_success``.
    4. Reset cached intermediates and cartesian error.
    5. Re-place object at (x, y, init_object_height) with the trajectory's
       canonical initial orientation; zero its velocity.
    6. Re-place robot at default joint positions (UR5 init + hand zeros + j12=0.5)
       and zero qvel.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # ---- 0) record per-episode pregrasp-success for the curriculum -----
    # Only envs that actually ran at least one step (``current_step > 0``)
    # contribute. This naturally skips the very first env-construction reset
    # (where every env has ``current_step == 0``) so the buffer doesn't get
    # polluted with a flush of structural zeros.
    if hasattr(env, "_record_episode_success"):
        ran = env.current_step[env_ids] > 0
        finished = env_ids[ran]
        if finished.numel() > 0:
            env._record_episode_success(finished, env._pregrasp_success[finished])

    # ---- 1) trajectory randomisation -----------------------------------
    static = env._traj_static
    # Read the live curriculum stage from the cfg every reset so promotions
    # by the auto-curriculum take effect on the next env that resets,
    # without needing to mutate ``EventTermCfg.params`` from the outside.
    stage = int(env.cfg.task.stage)
    x, y, theta = trajectory_module.apply_stage_randomisation(
        env._traj_buffers, static, env_ids, stage
    )

    # The buffers have been updated; sync the convenience aliases on the env.
    env._traj_obj_t = env._traj_buffers.obj_t
    env._traj_obj_q = env._traj_buffers.obj_q
    env._traj_jpos = env._traj_buffers.jpos
    env._traj_pregrasp = env._traj_buffers.pregrasp_jpos

    # ---- 2) final goal position ----------------------------------------
    env._traj_target_pos[env_ids] = env._traj_buffers.obj_t[env_ids, -1]
    env._init_object_height[env_ids] = env._traj_buffers.init_pos[env_ids, 2]

    # ---- 3) counters ---------------------------------------------------
    env.current_step[env_ids] = 0
    env._traj_step[env_ids] = 0
    env._pregrasp_success[env_ids] = False

    # ---- 4) reset cartesian error --------------------------------------
    # NOTE: do NOT touch ``env._reward_cache`` here. Reset events fire
    # *inside* ``super().step()`` (between reward.compute and the parent
    # returning), but ``manager_env.step`` reads the cache *after* the parent
    # returns to populate ``Metrics/`` in extras["log"]. Wiping the cache here
    # caused those metrics (hand_jpos_err, obj_lift, num_finger_contacts, ...)
    # to disappear from TensorBoard the moment any env in a rollout time-outs.
    # The cache is unconditionally invalidated at the end of ``manager_env.step``
    # so a stale cache will never leak into the next physics step.
    if hasattr(env, "_cartesian_error") and env._cartesian_error is not None:
        env._cartesian_error[env_ids] = 0.0

    # ---- 5) place the object -------------------------------------------
    obj = env.scene["object"]
    n = env_ids.shape[0]
    init_pos = env._traj_buffers.init_pos[env_ids]            # (n, 3) world frame
    # ViViDex picks the trajectory's frame-0 quaternion for the object init.
    init_quat = env._traj_buffers.obj_q[env_ids, 0]            # (n, 4) wxyz

    # IsaacLab convention: write_root_pose_to_sim takes (n, 7) = [pos, quat]
    # and uses *world* coordinates (env origins are added internally? Let's
    # check) - actually we need to express in env-local frame and let the
    # scene wrapper add env origins. The convention in ManagerBasedRLEnv is
    # that scene["object"] data is already in env-local; but write_root_pose_to_sim
    # expects WORLD coords, hence we must add env origins.
    env_origin = env.scene.env_origins[env_ids]               # (n, 3)
    root_pos_w = init_pos + env_origin
    root_pose = torch.cat([root_pos_w, init_quat], dim=-1)
    obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)

    zero_vel = torch.zeros((n, 6), device=env.device)
    obj.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    # ---- 6) reset robot ------------------------------------------------
    robot = env.scene["robot"]
    default_joint_pos = robot.data.default_joint_pos[env_ids]      # (n, 22)
    default_joint_vel = robot.data.default_joint_vel[env_ids]
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(default_joint_pos, env_ids=env_ids)
    # Reset root pose to the default world placement
    default_root_pose = robot.data.default_root_state[env_ids, :7].clone()
    default_root_pose[:, 0:3] += env_origin
    robot.write_root_pose_to_sim(default_root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=env_ids)
