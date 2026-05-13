"""Custom :class:`ManagerBasedRLEnv` subclass for the AllegroUR5 relocate task.

Two responsibilities beyond stock IsaacLab behaviour:

1. Build the YCB ``RigidObjectCfg`` dynamically from the chosen trajectory's
   ``object_name`` *before* ``super().__init__`` (otherwise ``cfg.validate``
   barfs on ``object: MISSING``). Conversion of the visual OBJ to USD uses
   :class:`MeshConverter` and is cached under ``cache/usd/ycb/<object>/``.

2. Allocate per-env trajectory and state buffers **before** the
   ObservationManager performs its dry-run on each ObsTerm. We do this in an
   override of :py:meth:`load_managers` (which the parent calls *after* the
   scene has been built and asset views initialised, but *before* the
   ObservationManager exists).

We also override :py:meth:`step` to:

* Increment per-env ``current_step`` and ``traj_step`` (only for envs that
  did not just reset).
* Invalidate the reward intermediates cache so the next step re-computes them.
"""

from __future__ import annotations

from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.converters import MeshConverter
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from . import trajectory as trajectory_module
from .env_paths import YCB_USD_CACHE, ensure_cache_dirs, ycb_visual_obj
from .manager_env_cfg import (
    AllegroRelocateManagerEnvCfg,
    FINGER_BODY_NAMES,
    PALM_BODY_NAME,
    ROBOT_BODY_NAMES,
)


class AllegroRelocateManagerEnv(ManagerBasedRLEnv):
    """Manager-based env for ViViDex-style YCB relocate."""

    cfg: AllegroRelocateManagerEnvCfg

    # ------------------------------------------------------------------
    # __init__: build object cfg dynamically from the chosen trajectory
    # ------------------------------------------------------------------

    def __init__(
        self,
        cfg: AllegroRelocateManagerEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ):
        # 1) Load trajectory (CPU/numpy) just to learn which YCB object to spawn.
        self._traj_static = trajectory_module.load_trajectory(cfg.task.trajectory_name)

        # 2) Build the per-env RigidObjectCfg for the YCB object.
        #    Pass cfg explicitly because ``self.cfg`` is only assigned by the
        #    base class ``__init__`` further down.
        cfg.scene.object = self._build_object_cfg(self._traj_static.object_name, cfg=cfg)

        # 3) Re-scale the PhysX broad-phase pairs capacity now that the user
        #    may have updated ``cfg.scene.num_envs``. The default in
        #    ``__post_init__`` is computed from the cfg-time num_envs (which
        #    is typically 64 in the cfg defaults), so we recompute here.
        #    Empirically PhysX requested up to ~17 pairs/env at 4096 envs, so
        #    we budget 64/env with a generous floor for robustness.
        required_pairs = max(16 * 1024, 64 * cfg.scene.num_envs)
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = max(
            cfg.sim.physx.gpu_total_aggregate_pairs_capacity, required_pairs
        )

        # 4) Initialise base env (creates scene, starts physics, calls load_managers()).
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    # ------------------------------------------------------------------
    # load_managers: pre-allocate buffers BEFORE the ObservationManager
    # does its dry-run.
    # ------------------------------------------------------------------

    def load_managers(self) -> None:  # type: ignore[override]
        # Resolve body / joint indices on the (already-initialised) Articulation.
        self._resolve_indices()
        # Allocate per-env buffers (trajectories + counters + state).
        self._allocate_buffers()
        # Now it is safe for the ObservationManager to query obs functions.
        super().load_managers()

    # ------------------------------------------------------------------
    # step: maintain counters and invalidate the reward cache.
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):  # type: ignore[override]
        ret = super().step(action)
        # Increment counters for envs that were *not* reset this step.
        not_reset = ~self.reset_buf.bool()
        self.current_step = torch.where(
            not_reset, self.current_step + 1, self.current_step
        )
        self._traj_step = torch.where(
            not_reset, self._traj_step + 1, self._traj_step
        )
        # Update pregrasp-success flag (vividex sets it once at the boundary).
        if hasattr(self, "_reward_cache") and self._reward_cache is not None:
            cache = self._reward_cache
            at_boundary = self.current_step == self._pregrasp_steps
            success_now = at_boundary & (~self._pregrasp_success) & (cache["pre_err"] < 0.05)
            self._pregrasp_success = self._pregrasp_success | success_now
        # Mirror vividex's diagnostic /info logs so they show up in TensorBoard.
        # rsl_rl's logger averages every key in extras["log"] across rollout steps
        # within an iteration, so writing per-step means here yields per-iteration
        # mean metrics under the ``Metrics/`` group.
        self._populate_metrics_log()
        # Invalidate the intermediates cache so the next step recomputes.
        self._reward_cache = None
        return ret

    def _populate_metrics_log(self) -> None:
        """Append vividex-style per-step diagnostic scalars to extras['log'].

        These keys mirror the ones vividex prints in its training table:

        * ``control_error``       -- DLS-IK Cartesian residual (m)
        * ``hand_jpos_err``       -- pre-grasp fingertip L2 error (m)
        * ``hand_mjpos_err``      -- imitate fingertip L2 error (m)
        * ``obj_com_err``         -- object COM tracking error (m)
        * ``obj_lift``            -- object height above init (m, clamped >=0)
        * ``pregrasp_success``    -- mean cumulative success bool across envs
        * ``imitate_steps``       -- per-env imitate horizon (constant in stages 0/1)
        * ``stage``               -- curriculum stage int (0/1/2)

        All values are scalars (mean across envs).  rsl_rl will then average
        them across rollout steps to produce one TB scalar per iteration.
        """

        if "log" not in self.extras:
            self.extras["log"] = {}
        log = self.extras["log"]

        cache = getattr(self, "_reward_cache", None)
        if cache is not None:
            log["Metrics/hand_jpos_err"] = cache["pre_err"].mean()
            log["Metrics/hand_mjpos_err"] = cache["fingertip_err"].mean()
            log["Metrics/obj_com_err"] = cache["obj_com_err"].mean()
            log["Metrics/obj_rot_err"] = cache["obj_rot_err"].mean()
            log["Metrics/obj_lift"] = cache["obj_lift"].mean()
            log["Metrics/num_finger_contacts"] = cache["num_finger_contacts"].mean()

        ce = getattr(self, "_cartesian_error", None)
        if ce is not None:
            log["Metrics/control_error"] = ce.mean()

        if hasattr(self, "_pregrasp_success"):
            log["Metrics/pregrasp_success"] = self._pregrasp_success.float().mean()

        if hasattr(self, "_imitate_steps"):
            log["Metrics/imitate_steps"] = self._imitate_steps.float().mean()

        # Stage is a static cfg int; cast to tensor for uniform tensor handling.
        try:
            stage = float(self.cfg.task.stage)
            log["Metrics/stage"] = torch.tensor(stage, device=self.device)
        except (AttributeError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_object_cfg(
        self,
        object_name: str,
        cfg: AllegroRelocateManagerEnvCfg | None = None,
    ) -> RigidObjectCfg:
        """Convert the YCB visual OBJ to USD (cached) and wrap in a
        :class:`RigidObjectCfg` with ``activate_contact_sensors=True``.
        """

        ensure_cache_dirs()
        # ViViDex uses two separate meshes per object:
        #   * visual/<obj>/textured_simple.obj   -- high-detail, with UVs/MTL
        #   * collision/<obj>/collision.obj      -- 2-group low-poly hull
        # IsaacLab's :class:`MeshConverter` only takes a single ``asset_path``
        # which is used for *both* rendering and the physics actor. We pick
        # the textured visual OBJ here (so the spawned USD ships with a
        # textured display mesh) and rely on PhysX's VHACD to decompose it
        # into ``max_convex_hulls=2`` convex shells for collision -- which
        # is what ViViDex's hand-authored ``collision.obj`` provides via its
        # 2 OBJ sub-groups.
        #
        # Trade-off vs feeding the explicit ``collision.obj`` to
        # ``MeshConverter``: VHACD on the 1.6 MB visual mesh produces 2
        # slightly different hulls than ViViDex's hand-authored 2-group OBJ,
        # which costs us about 0.04 mean return (3% relative). In return,
        # the rendered bottle shows the proper YCB texture instead of a
        # flat grey low-poly shell, which matters for video-based debugging
        # and any future vision-based policy.
        visual_obj_path = ycb_visual_obj(object_name)
        if not visual_obj_path.exists():
            raise FileNotFoundError(f"YCB visual OBJ not found: {visual_obj_path}")

        usd_dir = YCB_USD_CACHE / object_name
        usd_dir.mkdir(parents=True, exist_ok=True)
        # Mass is hard-coded to the YCB-published value (~0.603 kg for the
        # mustard bottle). PhysX' density x VHACD-volume product would be
        # off by ~50% otherwise, which throws off the friction budget the
        # ViViDex policy was trained against.
        #
        # ``max_convex_hulls=2`` is critical: with the default ~32-hull
        # decomposition the mustard bottle's interior edges and stair-stepped
        # surface let it squirt out of a 4-finger pinch grip. Capping at 2
        # hulls reproduces ViViDex's 2-group ``collision.obj`` topology.
        mesh_cfg = MeshConverterCfg(
            asset_path=str(visual_obj_path),
            usd_dir=str(usd_dir),
            usd_file_name="ycb.usd",
            mass_props=schemas_cfg.MassPropertiesCfg(mass=0.603),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(
                max_convex_hulls=2,
                voxel_resolution=200_000,
                shrink_wrap=True,
            ),
            make_instanceable=False,
        )
        converter = MeshConverter(mesh_cfg)
        usd_path = converter.usd_path

        # Optional grasp-stability tweaks, stashed on the task cfg's __dict__
        # by play_vividex.py (kept off the configclass schema on purpose so
        # they are pure runtime overrides). ``self.cfg`` is not yet bound
        # because the base class ``__init__`` runs after this helper, so the
        # caller passes ``cfg`` explicitly.
        task_cfg = cfg.task if cfg is not None else getattr(self, "cfg", None)
        task_dict = getattr(task_cfg, "__dict__", {})
        obj_iters = task_dict.get("_object_solver_iters")
        obj_max_depen = task_dict.get("_object_max_depen_vel")
        rigid_props_kwargs: dict[str, Any] = dict(
            disable_gravity=False,
            retain_accelerations=False,
        )
        if obj_iters is not None:
            rigid_props_kwargs["solver_position_iteration_count"] = int(obj_iters[0])
            rigid_props_kwargs["solver_velocity_iteration_count"] = int(obj_iters[1])
        if obj_max_depen is not None:
            rigid_props_kwargs["max_depenetration_velocity"] = float(obj_max_depen)

        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(**rigid_props_kwargs),
                # Re-affirm here in case the USD on disk was generated by
                # an older converter run with a different mass.
                mass_props=sim_utils.MassPropertiesCfg(mass=0.603),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.35, 0.35, max(self._traj_static.init_object_height, 0.05)),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

    def _resolve_indices(self) -> None:
        """Resolve body/joint indices on the (now-initialised) articulation."""

        robot = self.scene["robot"]

        # Joint links → 14 indices for ``robot_state``.
        body_idx_list, _ = robot.find_bodies(ROBOT_BODY_NAMES, preserve_order=True)
        if len(body_idx_list) != len(ROBOT_BODY_NAMES):
            raise RuntimeError(
                f"Could not resolve all 14 joint links. Wanted: {ROBOT_BODY_NAMES}; "
                f"available: {robot.body_names}"
            )
        self._joint_link_body_idx = torch.tensor(body_idx_list, device=self.device, dtype=torch.long)

        # Palm link.
        palm_ids, _ = robot.find_bodies([PALM_BODY_NAME], preserve_order=True)
        if len(palm_ids) != 1:
            raise RuntimeError(
                f"Could not resolve palm link '{PALM_BODY_NAME}'. Available: {robot.body_names}"
            )
        self._palm_body_idx = int(palm_ids[0])

        # 4 finger parent links.
        finger_ids, _ = robot.find_bodies(FINGER_BODY_NAMES, preserve_order=True)
        if len(finger_ids) != 4:
            raise RuntimeError(
                f"Could not resolve finger links {FINGER_BODY_NAMES}. Available: {robot.body_names}"
            )
        self._finger_body_idx = torch.tensor(finger_ids, device=self.device, dtype=torch.long)

        # ------------------------------------------------------------------
        # SAPIEN-compatible joint reordering.
        #
        # The ViViDex policy expects ``qpos`` in URDF declaration order:
        #   [arm_0..arm_5, joint_00, joint_01, ..., joint_15]
        # But IsaacLab's ``robot.data.joint_pos`` returns joints in articulation
        # internal order, which for this URDF is non-monotonic for the hand:
        #   [arm_0..arm_5, joint_00, joint_04, joint_08, joint_12, joint_01,
        #    joint_05, joint_09, joint_13, joint_02, joint_06, joint_10, ...].
        # We build an index tensor ``_qpos_sapien_idx`` that maps URDF position
        # k -> IsaacLab joint id, so ``joint_pos[:, _qpos_sapien_idx]`` matches
        # what SAPIEN's ``robot.get_qpos()`` would return to the policy at
        # training time. Consumed by ``mdp/observations.py:robot_state``.
        from .manager_env_cfg import ARM_JOINT_NAMES, HAND_JOINT_NAMES
        sapien_order_names = list(ARM_JOINT_NAMES) + list(HAND_JOINT_NAMES)
        sapien_ids, _ = robot.find_joints(sapien_order_names, preserve_order=True)
        if len(sapien_ids) != len(sapien_order_names):
            raise RuntimeError(
                "Could not resolve all 22 robot joints in SAPIEN URDF order"
            )
        self._qpos_sapien_idx = torch.tensor(
            sapien_ids, device=self.device, dtype=torch.long
        )

    def _allocate_buffers(self) -> None:
        """Pre-allocate per-env trajectory + state buffers."""

        n = self.num_envs
        device = self.device

        self._traj_buffers = trajectory_module.allocate_buffers(
            self._traj_static, n, device
        )
        # Convenience aliases (also updated in mdp/events.py).
        self._traj_obj_t = self._traj_buffers.obj_t
        self._traj_obj_q = self._traj_buffers.obj_q
        self._traj_jpos = self._traj_buffers.jpos
        self._traj_pregrasp = self._traj_buffers.pregrasp_jpos

        # Per-env target position (last frame of object_translation).
        self._traj_target_pos = self._traj_buffers.obj_t[:, -1].clone()
        # Per-env initial object height.
        self._init_object_height = torch.full(
            (n,), float(self._traj_static.init_object_height), device=device
        )

        # Counters / flags ------------------------------------------------
        self.current_step = torch.zeros(n, device=device, dtype=torch.long)
        self._traj_step = torch.zeros(n, device=device, dtype=torch.long)
        self._pregrasp_success = torch.zeros(n, device=device, dtype=torch.bool)
        self._pregrasp_steps = torch.full(
            (n,), int(self._traj_static.pregrasp_steps), device=device, dtype=torch.long
        )
        self._imitate_steps = torch.full(
            (n,), int(self._traj_static.imitate_steps), device=device, dtype=torch.long
        )

        # Reward intermediate cache placeholder.
        self._reward_cache = None
        # Cartesian error placeholder (updated each step by the action term).
        self._cartesian_error = torch.zeros(n, device=device)
        # Target linear velocity placeholder (set by IKHandAction every step).
        self._target_lin_vel = torch.zeros(n, 3, device=device)
