"""Custom 22-dim ``ActionTerm`` for the AllegroUR5 YCB relocate task.

Action layout (matches ViViDex)::

    action ∈ R^22, ∈ [-1, 1]
        [0:3]  palm linear  velocity command  (world frame, m/s)
        [3:6]  palm angular velocity command  (world frame, rad/s)
        [6:22] Allegro hand qpos targets       (16 joints, mapped from [-1,1] to joint limits)

The arm portion (first 6 entries) is integrated as a delta on the current
palm pose to obtain a target pose, fed through a :class:`DifferentialIKController`
to produce 6-d arm qpos targets. The hand portion (last 16) is mapped to
joint position targets directly.

We also expose a per-env scalar :attr:`cartesian_error` on the env (read by
the reward term ``controller_penalty``) and a :attr:`prev_palm_pos` buffer
used to compute that error, both updated inside :py:meth:`apply_actions`.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
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
    """Name of the palm rigid body (target frame for IK)."""

    cart_lin_vel_limit: float = 1.0
    """Maximum palm linear velocity (m/s) when the action is at ±1."""

    cart_ang_vel_limit: float = 1.0
    """Maximum palm angular velocity (rad/s) when the action is at ±1."""

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

        body_ids, body_names = self._asset.find_bodies(self.cfg.palm_body_name)
        if len(body_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one match for palm body '{self.cfg.palm_body_name}', got {body_names}."
            )
        self._palm_body_idx = body_ids[0]

        # Jacobian indexing convention (from IsaacLab DifferentialInverseKinematicsAction):
        # for fixed-base articulations, the jacobian skips the (non-existent) base body.
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._palm_body_idx - 1
            self._jacobi_arm_joint_ids = list(self._arm_joint_ids)
        else:
            self._jacobi_body_idx = self._palm_body_idx
            self._jacobi_arm_joint_ids = [i + 6 for i in self._arm_joint_ids]

        # ------------------------------------------------------------------
        # Differential IK controller (pose target, pose_abs)
        # ------------------------------------------------------------------
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": float(self.cfg.ik_damping)},
        )
        self._ik_controller = DifferentialIKController(
            cfg=ik_cfg, num_envs=self.num_envs, device=self.device
        )

        # ------------------------------------------------------------------
        # Buffers
        # ------------------------------------------------------------------
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_palm_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # cartesian error buffer (per env) -- exposed via env._cartesian_error
        self._cartesian_error = torch.zeros(self.num_envs, device=self.device)

        # Joint limits for the hand mapping (loaded once at init)
        # shape: (num_envs, num_joints)  -- IsaacLab gives per-env limits
        joint_limits = self._asset.data.soft_joint_pos_limits.clone()
        # We grab limits for the hand joints only -- shape (num_envs, 16, 2)
        self._hand_joint_lower = joint_limits[:, self._hand_joint_ids, 0]
        self._hand_joint_upper = joint_limits[:, self._hand_joint_ids, 1]

        # Bookkeeping: dt for the *control* step (= sim_dt * decimation)
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
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._target_lin_vel[env_ids] = 0.0
            self._target_ang_vel[env_ids] = 0.0
            self._cartesian_error[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        # ViViDex clips actions to [-1, 1].
        a = actions.clamp_(-1.0, 1.0)
        self._raw_actions[:] = a
        self._processed_actions[:] = a

        # ----- arm: integrate velocity to get target pose, then set IK ----
        self._target_lin_vel[:] = a[:, 0:3] * self.cfg.cart_lin_vel_limit
        self._target_ang_vel[:] = a[:, 3:6] * self.cfg.cart_ang_vel_limit

        palm_pos_w = self._asset.data.body_pos_w[:, self._palm_body_idx]
        palm_quat_w = self._asset.data.body_quat_w[:, self._palm_body_idx]
        # remember current palm position for the cartesian-error computation
        self._prev_palm_pos[:] = palm_pos_w

        # Integrate over a *control* step (decimation * sim_dt).
        target_pos_w = palm_pos_w + self._target_lin_vel * self._control_dt
        # Rotation from angular velocity * dt expressed as axis-angle
        delta_rot = self._target_ang_vel * self._control_dt  # (E, 3) axis-angle
        delta_quat = math_utils.quat_from_angle_axis(
            torch.linalg.norm(delta_rot, dim=-1).clamp(min=1e-9),
            torch.nn.functional.normalize(delta_rot, dim=-1, eps=1e-9),
        )
        target_quat_w = math_utils.quat_mul(delta_quat, palm_quat_w)

        command = torch.cat([target_pos_w, target_quat_w], dim=-1)  # (E, 7) pose_abs
        self._ik_controller.set_command(command, palm_pos_w, palm_quat_w)

        # Expose env-level state for reward terms / observation terms.
        # This mirrors vividex `cartesian_error` (computed at *next* step).
        self._env._target_lin_vel = self._target_lin_vel
        self._env._cartesian_error = self._cartesian_error

    def apply_actions(self) -> None:
        # ----- IK arm qpos target ----------------------------------------
        palm_pos_w = self._asset.data.body_pos_w[:, self._palm_body_idx]
        palm_quat_w = self._asset.data.body_quat_w[:, self._palm_body_idx]

        jacobian = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_arm_joint_ids
        ]
        arm_joint_pos = self._asset.data.joint_pos[:, self._arm_joint_ids]
        arm_joint_pos_des = self._ik_controller.compute(
            palm_pos_w, palm_quat_w, jacobian, arm_joint_pos
        )

        # ----- hand qpos target (linear map [-1, 1] → [lower, upper]) -----
        a_hand = self._processed_actions[:, 6:22]
        hand_joint_pos_des = (
            (a_hand + 1.0) * 0.5 * (self._hand_joint_upper - self._hand_joint_lower)
            + self._hand_joint_lower
        )

        # ----- write joint position targets in one call -------------------
        full_target = torch.cat([arm_joint_pos_des, hand_joint_pos_des], dim=-1)
        self._asset.set_joint_position_target(full_target, joint_ids=self._all_joint_ids)

        # Update cartesian error: ||(p_now - p_prev) - target_lin_vel * dt|| where
        # p_prev was captured in process_actions and p_now is the post-IK pose.
        # This matches vividex's definition of `cartesian_error`.
        # NOTE: this is computed once per *sim* step; we keep the latest value.
        with torch.no_grad():
            relative = palm_pos_w - self._prev_palm_pos
            expected = self._target_lin_vel * self._env.physics_dt
            self._cartesian_error[:] = torch.linalg.norm(relative - expected, dim=-1)
