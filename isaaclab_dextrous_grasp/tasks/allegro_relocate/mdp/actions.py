"""Custom 22-dim ``ActionTerm`` for the AllegroUR5 YCB relocate task.

Action layout (matches ViViDex)::

    action ∈ R^22, ∈ [-1, 1]
        [0:3]  palm linear  velocity command  (world frame, m/s)
        [3:6]  palm angular velocity command  (world frame, rad/s)
        [6:22] Allegro hand qpos targets       (16 joints, mapped from [-1,1] to joint limits)

This implementation follows ViViDex's
:py:meth:`AllegroRelocateRLEnv.arm_sim_step` exactly:

1. Compute ``arm_qvel = J⁺(q) · v_des`` via DLS in ``process_actions`` (once
   per outer step), where ``J`` is the palm-w.r.t.-arm-6-joints Jacobian.
2. Set ``arm_qpos_des = arm_qpos + arm_qvel · dt_ctrl``.
3. In ``apply_actions`` (called every sim sub-step), push **both** position
   and velocity targets to the implicit PD controller. The velocity
   feedforward is critical: without it the high-kd arm gains pin the joint
   to zero qvel and the IK solution is never realised.

We also expose a per-env scalar :attr:`cartesian_error` on the env (read by
the reward term ``controller_penalty``). It is updated every sub-step using
``step_dt`` so the *final* (after all decimation sub-steps) value matches
ViViDex's outer-step convention ``||Δpalm − v_des · dt_ctrl||``.
"""

from __future__ import annotations

import math as _math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class IKHandActionCfg(ActionTermCfg):
    """Configuration for :class:`IKHandAction`."""

    arm_joint_names: list[str] = MISSING
    """6 UR5 joint names, in order."""

    hand_joint_names: list[str] = MISSING
    """16 Allegro joint names, in order."""

    palm_body_name: str = MISSING
    """Name of the palm rigid body (used for diagnostics / observations)."""

    ik_body_name: str = "right_wrist_3_link"
    """Name of the body whose spatial Jacobian drives the arm IK.

    ViViDex's :class:`PartialKinematicModel` uses the *child link of the last
    arm joint* (i.e. ``right_wrist_3_link`` for the UR5 + Allegro setup), not
    the palm link. The variable is named ``palm_jacobian`` in
    ``hand_imitation/env/rl_env/base.py:166`` but is in fact the wrist-3
    Jacobian. The trained policy expects ``action[:6]`` to drive
    ``right_wrist_3_link``'s spatial velocity, so the IK target body must
    match. Keeping the palm link as the IK target adds a constant rigid
    offset (FT300 + mounting plate ≈ 18 cm in -y, 4.6 cm in -z) and biases
    the IK solution.
    """

    cart_lin_vel_limit: float = 1.0
    """Maximum linear velocity (m/s) of the IK body when ``action[:3] = ±1``."""

    cart_ang_vel_limit: float = 1.0
    """Maximum angular velocity (rad/s) of the IK body when ``action[3:6] = ±1``."""

    ik_damping: float = 0.05
    """Damped least-squares damping (matches ViViDex 0.05)."""

    def __post_init__(self):
        # set the class type
        self.class_type = IKHandAction


class IKHandAction(ActionTerm):
    """22-d mixed IK + hand action term."""

    cfg: IKHandActionCfg
    _asset: Articulation

    def __init__(self, cfg: IKHandActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # ------------------------------------------------------------------
        # Resolve joint and body indices
        # ------------------------------------------------------------------
        self._arm_joint_ids, _ = self._asset.find_joints(self.cfg.arm_joint_names, preserve_order=True)
        self._hand_joint_ids, _ = self._asset.find_joints(self.cfg.hand_joint_names, preserve_order=True)
        self._all_joint_ids = self._arm_joint_ids + self._hand_joint_ids
        if len(self._arm_joint_ids) != 6:
            raise RuntimeError(
                f"Expected 6 arm joints, got {len(self._arm_joint_ids)} for {self.cfg.arm_joint_names}"
            )
        if len(self._hand_joint_ids) != 16:
            raise RuntimeError(
                f"Expected 16 hand joints, got {len(self._hand_joint_ids)} for {self.cfg.hand_joint_names}"
            )

        palm_body_ids, palm_body_names = self._asset.find_bodies(self.cfg.palm_body_name)
        if len(palm_body_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one match for palm body '{self.cfg.palm_body_name}', got {palm_body_names}."
            )
        self._palm_body_idx = palm_body_ids[0]

        # IK target body. ViViDex uses ``right_wrist_3_link`` (child of the
        # last arm joint), NOT the palm. See :attr:`IKHandActionCfg.ik_body_name`.
        ik_body_ids, ik_body_names = self._asset.find_bodies(self.cfg.ik_body_name)
        if len(ik_body_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one match for IK body '{self.cfg.ik_body_name}', got {ik_body_names}."
            )
        self._ik_body_idx = ik_body_ids[0]

        # Jacobian indexing convention (from IsaacLab DifferentialInverseKinematicsAction):
        # for fixed-base articulations, the jacobian skips the (non-existent) base body.
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._ik_body_idx - 1
            self._jacobi_arm_joint_ids = list(self._arm_joint_ids)
        else:
            self._jacobi_body_idx = self._ik_body_idx
            self._jacobi_arm_joint_ids = [i + 6 for i in self._arm_joint_ids]

        # ------------------------------------------------------------------
        # Buffers
        # ------------------------------------------------------------------
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        # Snapshot of the IK body's position at the start of the control step
        # (used by the cartesian-error diagnostic). Tracks ``ik_body``, NOT the
        # palm, so the error matches ViViDex's ``ee_link`` convention.
        self._prev_ik_body_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # Cartesian error buffer (per env) -- exposed via env._cartesian_error.
        self._cartesian_error = torch.zeros(self.num_envs, device=self.device)
        # Frozen targets computed once per outer step in process_actions, then
        # pushed to PhysX every sub-step in apply_actions.
        self._arm_qpos_des = torch.zeros(self.num_envs, 6, device=self.device)
        self._arm_qvel_des = torch.zeros(self.num_envs, 6, device=self.device)
        self._hand_qpos_des = torch.zeros(self.num_envs, 16, device=self.device)
        # Identity matrix used in the DLS solve.
        self._eye6 = (
            torch.eye(6, device=self.device).unsqueeze(0).expand(self.num_envs, 6, 6)
        )
        self._lambda_sq: float = float(self.cfg.ik_damping) ** 2
        # ViViDex clips arm_qvel to [-π, +π] (see relocate_env.py:168).
        self._arm_qvel_clip: float = _math.pi

        # Joint limits for the hand mapping (loaded once at init).
        # shape: (num_envs, num_joints) -- IsaacLab gives per-env limits.
        joint_limits = self._asset.data.soft_joint_pos_limits.clone()
        # We grab limits for the hand joints only -- shape (num_envs, 16).
        self._hand_joint_lower = joint_limits[:, self._hand_joint_ids, 0]
        self._hand_joint_upper = joint_limits[:, self._hand_joint_ids, 1]

        # Bookkeeping: dt for the *control* step (= sim_dt * decimation).
        # ``self._env`` has both ``physics_dt`` and ``step_dt``.
        self._control_dt: float = float(self._env.step_dt)

    # ----------------------------------------------------------------------
    # ActionTerm API
    # ----------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 22

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._target_lin_vel.zero_()
            self._target_ang_vel.zero_()
            self._cartesian_error.zero_()
            self._arm_qpos_des.zero_()
            self._arm_qvel_des.zero_()
            self._hand_qpos_des.zero_()
            self._prev_ik_body_pos.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._target_lin_vel[env_ids] = 0.0
            self._target_ang_vel[env_ids] = 0.0
            self._cartesian_error[env_ids] = 0.0
            self._arm_qpos_des[env_ids] = 0.0
            self._arm_qvel_des[env_ids] = 0.0
            self._hand_qpos_des[env_ids] = 0.0
            self._prev_ik_body_pos[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Internal: damped-least-squares IK (matches ViViDex's get_arm_qvel).
    # Solves:    Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ v_des
    # ------------------------------------------------------------------
    def _dls_ik(self, jacobian: torch.Tensor, v_des: torch.Tensor) -> torch.Tensor:
        # jacobian: (E, 6, 6); v_des: (E, 6) -> (E, 6)
        JJT = jacobian @ jacobian.transpose(-1, -2)
        A = JJT + self._lambda_sq * self._eye6
        rhs = torch.linalg.solve(A, v_des.unsqueeze(-1))
        dq = jacobian.transpose(-1, -2) @ rhs
        return dq.squeeze(-1)

    def process_actions(self, actions: torch.Tensor) -> None:
        # ViViDex clips actions to [-1, 1].
        a = actions.clamp_(-1.0, 1.0)
        self._raw_actions[:] = a
        self._processed_actions[:] = a

        # ----- arm: derive qvel via DLS, integrate over the control step ---
        self._target_lin_vel[:] = a[:, 0:3] * self.cfg.cart_lin_vel_limit
        self._target_ang_vel[:] = a[:, 3:6] * self.cfg.cart_ang_vel_limit
        twist = torch.cat([self._target_lin_vel, self._target_ang_vel], dim=-1)  # (E, 6)

        # Snapshot IK-body pose for the cartesian-error computation later.
        # ViViDex uses ``ee_link.get_pose().p`` (= wrist_3) at the start of
        # the outer step; we mirror that here.
        ik_pos_w = self._asset.data.body_pos_w[:, self._ik_body_idx]
        self._prev_ik_body_pos[:] = ik_pos_w

        # Jacobian at current configuration.
        jacobian = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_arm_joint_ids
        ]  # (E, 6, 6)

        arm_qvel = self._dls_ik(jacobian, twist)
        # ViViDex clips per-joint velocity to [-π, +π] (relocate_env.py:168).
        arm_qvel = arm_qvel.clamp_(-self._arm_qvel_clip, self._arm_qvel_clip)

        arm_qpos = self._asset.data.joint_pos[:, self._arm_joint_ids]
        # Frozen targets for all decimation sub-steps.
        self._arm_qpos_des[:] = arm_qpos + arm_qvel * self._control_dt
        self._arm_qvel_des[:] = arm_qvel

        # ----- hand qpos target (linear map [-1, 1] → [lower, upper]) -----
        a_hand = a[:, 6:22]
        self._hand_qpos_des[:] = (
            (a_hand + 1.0) * 0.5 * (self._hand_joint_upper - self._hand_joint_lower)
            + self._hand_joint_lower
        )

        # Expose env-level state for reward terms / observation terms.
        self._env._target_lin_vel = self._target_lin_vel
        self._env._cartesian_error = self._cartesian_error

    def apply_actions(self) -> None:
        # Push the (frozen) per-step targets to the implicit PD controller every
        # sim sub-step. This mirrors ViViDex, which calls
        # ``set_drive_target / set_drive_velocity_target`` once per outer step
        # and then steps the simulator ``frame_skip`` times.
        full_pos = torch.cat([self._arm_qpos_des, self._hand_qpos_des], dim=-1)
        self._asset.set_joint_position_target(full_pos, joint_ids=self._all_joint_ids)

        # Velocity feedforward: arm only (hand → 0).
        full_vel = torch.zeros(
            (self.num_envs, len(self._all_joint_ids)), device=self.device
        )
        full_vel[:, : len(self._arm_joint_ids)] = self._arm_qvel_des
        self._asset.set_joint_velocity_target(full_vel, joint_ids=self._all_joint_ids)

        # Update cartesian error every sub-step using ``step_dt`` so that the
        # final value (after all decimation sub-steps) matches ViViDex's
        # outer-step convention ``||Δee − v_des · dt_ctrl||`` where
        # ``ee = right_wrist_3_link``.
        with torch.no_grad():
            ik_pos_w = self._asset.data.body_pos_w[:, self._ik_body_idx]
            relative = ik_pos_w - self._prev_ik_body_pos
            expected = self._target_lin_vel * self._control_dt
            self._cartesian_error[:] = torch.linalg.norm(relative - expected, dim=-1)
