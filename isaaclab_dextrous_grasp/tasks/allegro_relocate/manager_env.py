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
        # We use ``>=`` instead of ``==`` so a single missed cache update can't
        # leave a successful env permanently flagged as failure (e.g. if the
        # cache happens to be wiped in the same step the env reaches the
        # boundary, or if curriculum-triggered force-resets land on that
        # exact step boundary).
        if hasattr(self, "_reward_cache") and self._reward_cache is not None:
            cache = self._reward_cache
            at_or_past_boundary = self.current_step >= self._pregrasp_steps
            success_now = at_or_past_boundary & (~self._pregrasp_success) & (cache["pre_err"] < 0.05)
            self._pregrasp_success = self._pregrasp_success | success_now
        # Mirror vividex's diagnostic /info logs so they show up in TensorBoard.
        # rsl_rl's logger averages every key in extras["log"] across rollout steps
        # within an iteration, so writing per-step means here yields per-iteration
        # mean metrics under the ``Metrics/`` group.
        self._populate_metrics_log()
        # Auto-curriculum: maybe promote ``cfg.task.stage`` and force-reset.
        # Must happen *after* metrics so the (pre-promotion) success rate is
        # what TB sees for this iteration.
        self._maybe_promote_stage()
        # Invalidate the intermediates cache so the next step recomputes.
        self._reward_cache = None
        return ret

    def _populate_metrics_log(self) -> None:
        """Append vividex-style per-step diagnostic scalars to extras['log'].

        These keys mirror the ones vividex prints in its training table:

        * ``control_error``             -- DLS-IK Cartesian residual (m)
        * ``hand_jpos_err``             -- pre-grasp fingertip L2 error (m)
        * ``hand_mjpos_err``            -- imitate fingertip L2 error (m)
        * ``obj_com_err``               -- object COM tracking error (m)
        * ``obj_lift``                  -- object height above init (m, clamped >=0)
        * ``pregrasp_success``          -- per-episode success rate (vividex Monitor)
        * ``pregrasp_success_timewise`` -- legacy per-step mean of the sticky bool
        * ``imitate_steps``             -- per-env imitate horizon (constant in stages 0/1)
        * ``stage``                     -- curriculum stage int (0/1/2)

        All values are scalars (mean across envs).  rsl_rl will then average
        them across rollout steps to produce one TB scalar per iteration.

        ``pregrasp_success`` semantics
        ------------------------------
        ViViDex's ``Monitor`` reports "fraction of the most recent ``N``
        finished episodes that flagged ``pregrasp_success`` before
        terminating", which sits in [0, 1] regardless of episode length and
        is what their auto-curriculum compares against ``0.95``.  Our
        ``_episode_success_history`` ring buffer stores exactly that signal
        (one entry per finished episode, pushed by ``mdp/events.py`` right
        before the flag is wiped).  The legacy ``_pregrasp_success`` mean
        across (env, step) pairs caps at ``pregrasp_steps / imitate_steps``
        ≈ ``0.75`` for healthy training because the flag is only ``True``
        for ``imitate_steps - pregrasp_steps`` of every episode -- we keep
        it under ``pregrasp_success_timewise`` for backward-compat with
        older TB plots / docs.
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
            log["Metrics/pregrasp_success_timewise"] = self._pregrasp_success.float().mean()

        # Per-episode success rate (vividex Monitor semantics).
        if getattr(self, "_episode_success_count", 0) > 0:
            valid = self._episode_success_history[: self._episode_success_count]
            log["Metrics/pregrasp_success"] = valid.float().mean()
        elif hasattr(self, "_pregrasp_success"):
            # Buffer empty (e.g. first iteration before any episode has
            # finished). Fall back to the timewise value so the TB curve
            # isn't NaN -- this is just a warm-up artefact.
            log["Metrics/pregrasp_success"] = self._pregrasp_success.float().mean()

        if hasattr(self, "_imitate_steps"):
            log["Metrics/imitate_steps"] = self._imitate_steps.float().mean()

        # Stage may be promoted at runtime by the auto-curriculum; read it
        # live from the cfg (which is what events.py also consumes).
        try:
            stage = float(self.cfg.task.stage)
            log["Metrics/stage"] = torch.tensor(stage, device=self.device)
        except (AttributeError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Curriculum: rolling per-episode success rate and stage promotion
    # ------------------------------------------------------------------

    def _record_episode_success(self, env_ids: torch.Tensor, success: torch.Tensor) -> None:
        """Push one bool per finished env into the rolling success buffer.

        Called by :func:`mdp.events.reset_trajectory_state` before
        ``_pregrasp_success[env_ids]`` is wiped. Buffer is a 1-D ring of
        capacity ``cfg.task.curriculum_history_size`` (default ``num_envs``).
        """

        n = int(env_ids.shape[0])
        if n == 0 or not hasattr(self, "_episode_success_history"):
            return
        cap = self._episode_success_history.shape[0]
        idx = int(self._episode_success_idx)
        vals = success.to(dtype=self._episode_success_history.dtype)
        end = idx + n
        if end <= cap:
            self._episode_success_history[idx:end] = vals
        else:
            first = cap - idx
            self._episode_success_history[idx:] = vals[:first]
            self._episode_success_history[: end - cap] = vals[first:]
        self._episode_success_idx = end % cap
        self._episode_success_count = min(self._episode_success_count + n, cap)
        self._episodes_completed_total += n

    def _curriculum_history_rate(self) -> float:
        """Return the mean of the rolling per-episode success buffer."""

        cnt = int(getattr(self, "_episode_success_count", 0))
        if cnt == 0:
            return 0.0
        return float(self._episode_success_history[:cnt].float().mean().item())

    def _maybe_promote_stage(self) -> None:
        """If the rolling success rate clears the threshold, bump the stage.

        Mirrors ViViDex's ``Monitor`` based promotion (``>= 0.95`` over the
        last ``num_envs`` finished episodes) and force-resets every env so
        the new stage's randomisation kicks in immediately. The history
        buffer is wiped so the next promotion is judged on fresh evidence.
        """

        task_cfg = getattr(self.cfg, "task", None)
        if task_cfg is None or not getattr(task_cfg, "auto_curriculum", False):
            return
        max_stage = int(getattr(task_cfg, "curriculum_max_stage", 2))
        cur_stage = int(task_cfg.stage)
        if cur_stage >= max_stage:
            return
        cnt = int(getattr(self, "_episode_success_count", 0))
        min_eps = int(getattr(task_cfg, "curriculum_min_episodes", 0)) or self.num_envs
        if cnt < min_eps:
            return
        threshold = float(getattr(task_cfg, "curriculum_threshold", 0.95))
        rate = self._curriculum_history_rate()
        if rate < threshold:
            return

        new_stage = cur_stage + 1
        print(
            f"[CURRICULUM] stage {cur_stage} -> {new_stage} "
            f"(rate={rate:.4f} over {cnt} eps, threshold={threshold:.2f})",
            flush=True,
        )
        # 1) update the cfg (events.py reads ``env.cfg.task.stage`` live).
        task_cfg.stage = new_stage
        # 2) wipe the rolling buffer so the next promotion is judged on
        #    new-stage evidence only.
        self._episode_success_history.zero_()
        self._episode_success_idx = 0
        self._episode_success_count = 0
        # 3) force-reset every env so the new (x,y,theta) randomisation
        #    takes effect on the next physics step rather than waiting for
        #    natural episode ends. ``_reset_idx`` re-runs all reset events,
        #    including ``reset_trajectory_state`` which now sees the new
        #    stage. We skip the per-episode bookkeeping the buffer would
        #    normally do for these resets (they are "structural", not
        #    real episode endings) by zeroing ``current_step`` first --
        #    ``reset_trajectory_state`` only records envs whose
        #    ``current_step > 0``.
        self.current_step.zero_()
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_env_ids)
        # NOTE: the obs returned to PPO from this step's ``super().step()``
        # is already stale w.r.t. the freshly reset state, but the next
        # ``super().step()`` will write the reset joint targets to PhysX in
        # its decimation loop and recompute observations -- this only costs
        # one slightly-stale transition at the moment of promotion, which
        # is rare (≤ 2 events per training run) and acceptable.

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

        # ---- curriculum: rolling per-episode pregrasp-success buffer ----
        # Sized after ``cfg.task.curriculum_history_size`` (defaults to
        # ``num_envs``, matching vividex's Monitor window of the most recent
        # 4096 episodes when training with 4096 envs).
        hist_size = int(getattr(self.cfg.task, "curriculum_history_size", 0)) or n
        self._episode_success_history = torch.zeros(hist_size, device=device, dtype=torch.float32)
        self._episode_success_idx = 0
        self._episode_success_count = 0
        self._episodes_completed_total = 0
