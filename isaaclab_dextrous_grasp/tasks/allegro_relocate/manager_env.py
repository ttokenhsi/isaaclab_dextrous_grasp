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
        cfg.scene.object = self._build_object_cfg(self._traj_static.object_name)

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
        # Invalidate the intermediates cache so the next step recomputes.
        self._reward_cache = None
        return ret

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_object_cfg(self, object_name: str) -> RigidObjectCfg:
        """Convert the YCB visual OBJ to USD (cached) and wrap in a
        :class:`RigidObjectCfg` with ``activate_contact_sensors=True``.
        """

        ensure_cache_dirs()
        obj_obj_path = ycb_visual_obj(object_name)
        if not obj_obj_path.exists():
            raise FileNotFoundError(f"YCB visual OBJ not found: {obj_obj_path}")

        usd_dir = YCB_USD_CACHE / object_name
        usd_dir.mkdir(parents=True, exist_ok=True)
        # Run the OBJ → USD conversion now (cheap if the cache is valid).
        # Mass is computed from density (matches ViViDex
        # ``ycb_object_utils.py:118-122`` which sets ``density=1000`` and
        # lets SAPIEN compute the per-object mass from the convex
        # decomposition volume). Hard-coding mass=0.2 kg made objects
        # ~5× lighter than the real YCB items (mustard bottle ≈ 1 kg)
        # and they would fly off on the slightest finger contact.
        mesh_cfg = MeshConverterCfg(
            asset_path=str(obj_obj_path),
            usd_dir=str(usd_dir),
            usd_file_name="ycb.usd",
            mass_props=schemas_cfg.MassPropertiesCfg(density=1000.0),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
            make_instanceable=False,
        )
        converter = MeshConverter(mesh_cfg)
        usd_path = converter.usd_path

        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                ),
                # Density-based; the converter already wrote density=1000
                # into the USD asset, but we re-affirm here in case the
                # USD on disk was generated by an older converter run.
                mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
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
