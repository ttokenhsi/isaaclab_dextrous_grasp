# ViViDex (SAPIEN) → IsaacLab Migration Notes

This document records the design, alignment and pitfalls of porting the
[ViViDex](https://github.com/zerchen/vividex_sapien) UR5 + Allegro YCB
relocate task from SAPIEN to IsaacLab's manager-based RL framework. It is
intentionally self-contained so that the
`/root/workspace/rl_grasp/isaaclab_dextrous_grasp` package can be vendored
without the original `vividex_sapien` repository.

The IsaacLab implementation lives in
`isaaclab_dextrous_grasp/tasks/allegro_relocate/` and is registered with
`gym.register("Isaac-AllegroUR5-Relocate-v0", ...)`.

---

## 1. High-level mapping

| ViViDex / SAPIEN                                                       | IsaacLab equivalent                                                                                  |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `AllegroRelocateRLEnv` (gym + SAPIEN actor system)                     | `AllegroRelocateManagerEnv` ⊂ `ManagerBasedRLEnv`                                                    |
| YAML / Hydra config (`config/agent/ppo.yaml`)                          | `@configclass` Python configs in `manager_env_cfg.py` + `agents/rsl_rl_ppo_cfg.py`                   |
| `frame_skip = 10`, `sim_freq = 500`                                    | `decimation = 10`, `sim.dt = 1/200` ⇒ effective 50 Hz control                                        |
| `compute_inverse_kinematics(...)` w/ damping=0.05 cart limit=1.0       | `IKHandAction` w/ `DifferentialIKController(command_type="velocity")`                                |
| Palm vel + 16 hand qpos action `(22,)`                                 | `IKHandActionCfg` (palm 6-d + Allegro 16-d) → `articulation.set_joint_position_target`               |
| 393-dim oracle observation                                             | 4 `ObsTerm` functions concatenated under the `policy` group                                          |
| Hand-coded reward inside `get_reward`                                  | 7 `RewTerm` (pregrasp / contact / object_track / fingertip_track / lift / ctrl / action)             |
| `is_done` flags                                                        | 4 `DoneTerm` (`pregrasp_failure`, `object_too_far`, `lost_contact_in_imitate`, `time_out`)           |
| `reset` rebuilds trajectory via `randomize_trajectories`               | `EventTerm(mode="reset")` → `reset_trajectory_state`                                                 |
| `check_actor_pair_contacts(palm, finger_parents, object)`              | 5 × `ContactSensorCfg` filtered to `{ENV_REGEX_NS}/Object`                                           |
| `stable_baselines3.PPO`                                                | `rsl_rl.runners.OnPolicyRunner` (rsl-rl-lib ≥ 5.0.0)                                                 |
| Single-env stepping × `n_envs` Python procs                            | Vectorised stepping by `InteractiveScene` (`num_envs` envs in one process)                           |

---

## 2. Action space (22 dim, `∈ [-1, 1]`)

| Slice | Meaning                                          | ViViDex source                                            |
| ----- | ------------------------------------------------ | --------------------------------------------------------- |
| `[0:3]`  | Palm world-frame **linear** velocity command  | `vividex_sapien/.../base.py::compute_inverse_kinematics` |
| `[3:6]`  | Palm world-frame **angular** velocity command | same                                                      |
| `[6:22]` | Allegro 16-DoF joint position targets         | `relocate_env.py::pre_step`                               |

`IKHandAction` (in `mdp/actions.py`):

1. Scales `[-1, 1]` to physical limits (`cart_lin_vel_limit=1.0`,
   `cart_ang_vel_limit=1.0`, Allegro per-joint `joint_limits`).
2. Feeds the 6-DoF palm twist to a `DifferentialIKController`
   (`command_type="velocity"`, damping `0.05`) whose Jacobian is taken from
   `right_gripper_palm_link`. This yields 6 arm joint deltas per env.
3. Adds the deltas to the current arm qpos to obtain absolute targets, and
   concatenates with the 16 hand qpos targets.
4. Applies `articulation.set_joint_position_target(targets)`.

Two byproducts are cached on the action term for use by reward terms:

- `cartesian_error`: `‖actual_palm_lin_vel − target_lin_vel‖²` per env.
- `target_lin_vel`: the unscaled 3-d palm linear velocity command.

This matches `compute_inverse_kinematics` exactly in geometry (Jacobian
solve in world frame, identical damping) and timing (one IK solve per
50 Hz outer step, `decimation=10` PhysX sub-steps).

---

## 3. Observation space (393 dim oracle)

All four observation terms live under the `policy` observation group and
share the same intermediate buffers via the action term and the
trajectory buffers in `AllegroRelocateManagerEnv`.

| Term                | Dim | Composition                                                                                                                                     |
| ------------------- | --- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `robot_state`       | 330 | qpos (22) + qvel (22) + 22 link states; the 22 links concatenate **block-wise**: 22×3 pos, 22×4 quat, 22×3 lin-vel, 22×3 ang-vel = 286 dims     |
| `object_state`      | 13  | object pos (3) + quat (4) + lin-vel (3) + ang-vel (3)                                                                                           |
| `goal_state`        | 42  | next-3-frame `[orn(4) + trans(3)]` = 21 + palm−object diff (3) + 4 fingertip−object diffs (12) + palm−target diff (3) + object−target diff (3)  |
| `time_state`        | 8   | `[sin(k·t), cos(k·t)]` for `k ∈ {1, 4, 6, 8}`, with `t = traj_step / imitate_steps`                                                             |

`ROBOT_BODY_NAMES` in `manager_env_cfg.py` lists the 22 links the
ViViDex `joint_link_names` references: 6 arm links
(`shoulder_link`, `upper_arm_link`, …, `right_gripper_palm_link`) plus the
16 Allegro distal/medial links. The block-wise flattening (`pos|quat|lv|av`
across the whole batch) is a critical detail; the older "interleaved"
flattening of `(pos, quat, lv, av)` per link gave 393 dims as well, **but
in a different order**, breaking sample efficiency and aligning poorly
with the loaded ViViDex policies.

The `oracle_state` group is also exposed for the critic via the
`obs_groups = {"actor": ["policy"], "critic": ["policy"]}` mapping,
matching the original PPO setup that uses identical observations for
both networks.

---

## 4. Reward (matches ViViDex `get_reward / 10`)

The seven `RewTerm`s are defined in `mdp/rewards.py`. Their per-step
values are the exact terms from ViViDex's `get_reward`; the global
`/10` factor is folded into the term `weight`s declared in
`manager_env_cfg.py`. This way the manager log keeps the reward
sub-components legible.

| RewTerm                 | weight | ViViDex term                                                                                |
| ----------------------- | ------ | ------------------------------------------------------------------------------------------- |
| `pregrasp`              | 1.0    | `10 · exp(-10 · fingertip_err)` while `step ≤ pregrasp_steps`, else 0                       |
| `contact`               | 0.05   | `0.5 · num_finger_contacts` while `step > pregrasp_steps`, else 0                           |
| `object_track`          | 1.0    | `10 · exp(-50 · (com_err + 0.1 · rot_err))` while imitating                                 |
| `fingertip_track`       | 0.4    | `4 · exp(-10 · fingertip_err)` while imitating                                              |
| `lift_bonus`            | 0.25   | `2.5 · 1{lift > 0.02}` while imitating                                                      |
| `controller_penalty`    | -100.0 | `-1e3 · cartesian_error²`                                                                   |
| `action_penalty`        | -0.001 | `-0.01 · ∑ clip(qvel, -1, 1)²`                                                              |

Intermediates (`com_err`, `rot_err`, `fingertip_err`, palm/finger contact
booleans, `lift`) are computed once per step in `_ensure_intermediates`
and cached on the env. The `step()` override in `AllegroRelocateManagerEnv`
invalidates the cache on each environment step so reward / termination /
observations always see consistent values for the same physics frame.

### Contact term in detail

`num_finger_contacts` is the bucket count, exactly as in ViViDex
(`check_actor_pair_contacts(palm + 4 finger parents, object)`). We obtain
it from 5 `ContactSensor`s (one per bucket) and threshold the
`force_matrix_w` magnitude with `impulse_threshold = 1e-2 / dt`. The
`force_matrix_w` is filtered to a single body per env (`{ENV_REGEX_NS}/Object`)
which gives a deterministic shape across all envs.

---

## 5. Termination (4 DoneTerms)

| DoneTerm                  | Condition                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| `pregrasp_failure`        | `step == pregrasp_steps + 1` and `‖hand_qpos − ref_hand_qpos‖ > 0.05`                    |
| `object_too_far`          | `‖object_pos − ref_object_pos‖ > 0.15`                                                   |
| `lost_contact_in_imitate` | `step > pregrasp_steps` and all 5 contact buckets empty                                  |
| `time_out`                | `episode_length_buf ≥ max_episode_length` (sets `truncated=True`, not `terminated`)      |

Mirrors `is_done` in `AllegroRelocateRLEnv` exactly. `time_out=True` is
the IsaacLab way to expose Gymnasium's truncation flag.

---

## 6. Events (resets)

`mdp/events.py::reset_trajectory_state` is registered as
`EventTerm(mode="reset")`. For each env that needs to reset, it:

1. Samples a trajectory id (according to `task.trajectory_names`).
2. Picks an in-plane offset / yaw rotation according to `task.stage`
   (0 = canonical, 1 = `(x,y) ∈ U[0.30, 0.40]²`, 2 = +`θ_z ∈ U(-π/12, π/12)`).
3. Vectorises the trajectory canonicalisation (`object_translation`,
   `object_orientation`, `robot_jpos`) **identically** to ViViDex's
   `randomize_trajectories`.
4. Writes the canonical reference into per-env buffers
   (`env._traj_object_pos`, `env._traj_robot_qpos`, …).
5. Resets `current_step / traj_step / pregrasp_failure_pending` counters.
6. Sets the robot articulation state and the YCB object's root state to
   the trajectory's t=0 frame.

The buffers themselves are allocated once in
`AllegroRelocateManagerEnv.load_managers` (see §8), before the parent's
manager construction, so that the obs/reward/done callbacks find them
populated when they run their dry-run.

---

## 7. Curriculum (`task.stage`)

Mirrors ViViDex's stages exactly. The rotations are applied to the
trajectory **and** the object's initial pose in
`reset_trajectory_state`, so that the hand reference still matches the
randomised object:

```
stage 0 → (x, y, θz) = (0.35, 0.35, 0)
stage 1 → x, y ∼ U[0.30, 0.40], θz = 0
stage 2 → x, y ∼ U[0.30, 0.40], θz ∼ U(-π/12, +π/12)
```

---

## 8. Custom env override (`AllegroRelocateManagerEnv`)

The default `ManagerBasedRLEnv` is too rigid for two reasons; the
subclass in `manager_env.py` patches them:

1. **Object asset is data-driven.** ViViDex picks a YCB object based
   on the trajectory's `object_name`. We replicate this by **mutating
   `cfg.scene.object` in `__init__` before calling `super().__init__`**:
   - Look up the OBJ path under `assets/ycb/<id>/textured_simple.obj`.
   - Convert to USD via `MeshConverter` (cached in `cache/usd`).
   - Build a `RigidObjectCfg` with `activate_contact_sensors=True` so the
     contact sensor pipeline can attach to the object body.

2. **`ObsTerm` dry-run requires the trajectory buffers up-front.** When
   IsaacLab's `ObservationManager` is constructed it calls every
   observation function once with synthetic data to infer shapes. Our
   `goal_state` reads `env._traj_object_pos[:, env.traj_step + Δ]`, which
   would crash. We therefore override `load_managers` to:
   1. Resolve articulation body/joint indices.
   2. Allocate `_traj_*` and counter buffers with sensible defaults.
   3. Then call `super().load_managers()`.

3. **Per-step bookkeeping**: `step()` is overridden to:
   - Increment `current_step` and `traj_step` (`min(current_step, T-1)`).
   - Invalidate the `_intermediate_cache` so reward / termination always
     re-derive intermediates against the post-step state.

---

## 9. Contact sensors

We follow IsaacLab's recommended three-step recipe:

1. **Robot URDF spawn** with `activate_contact_sensors=True` so the
   `right_gripper_palm_link` and finger parent links expose the
   contact-reporter API.
2. **Object spawn** with `activate_contact_sensors=True` so the YCB body
   counts as a valid contact partner.
3. **5 `ContactSensorCfg`** in the scene config bound to:
   - `right_gripper_palm_link` (palm, bucket 0)
   - `right_gripper_link_15` (thumb parent, bucket 1)
   - `right_gripper_link_03` (index parent, bucket 2)
   - `right_gripper_link_07` (middle parent, bucket 3)
   - `right_gripper_link_11` (ring parent, bucket 4)
   Each has `filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]` so the
   `force_matrix_w` shape is a deterministic `(num_envs, num_bodies, 1, 3)`.

`num_finger_contacts` per env is the count of buckets whose force
magnitude exceeds `1e-2 / dt`. This is **the** quantity the contact
reward and the `lost_contact_in_imitate` termination consume, and it is
formally identical to ViViDex's
`check_actor_pair_contacts(...)` summed over the four finger buckets.

---

## 10. PPO config (`agents/rsl_rl_ppo_cfg.py`)

Aligned with `vividex_sapien/algos/rl/config/agent/ppo.yaml`:

| Field                | ViViDex          | IsaacLab equivalent (rsl-rl 5.x)                              |
| -------------------- | ---------------- | ------------------------------------------------------------- |
| `gamma`              | 0.95             | `RslRlPpoAlgorithmCfg.gamma=0.95`                             |
| `gae_lambda`         | 0.95             | `lam=0.95`                                                    |
| `learning_rate`      | 1e-5 (fixed)     | `learning_rate=1e-5`, `schedule="fixed"`                      |
| `ent_coef`           | 0.001            | `entropy_coef=0.001`                                          |
| `vf_coef`            | 0.5              | `value_loss_coef=0.5`                                         |
| `clip_range`         | 0.2              | `clip_param=0.2`, `use_clipped_value_loss=True`               |
| `n_steps` (per env)  | 4096 / num_envs  | `num_steps_per_env=64` (default num_envs=64 ⇒ 4096)           |
| `batch_size` (mini)  | 256              | `num_mini_batches = (64 × 64) / 256 = 16`                     |
| `n_epochs`           | 5                | `num_learning_epochs=5`                                       |
| `net_arch=[256,128]` | actor and critic | `RslRlMLPModelCfg(hidden_dims=[256,128], activation="elu")`   |
| `log_std_init=-1.6`  | exp(-1.6)≈0.20   | `GaussianDistributionCfg(init_std=0.20, std_type="scalar")`   |
| `desired_kl=0.01`    | n/a              | early-stopping signal (rsl-rl extra)                          |
| `max_grad_norm=1.0`  | n/a              | gradient clipping (rsl-rl extra)                              |

For rsl-rl ≥ 5.0.0 the legacy `policy=...` field is gone; we use separate
`actor: RslRlMLPModelCfg` and `critic: RslRlMLPModelCfg` and the
`obs_groups={"actor": ["policy"], "critic": ["policy"]}` mapping. The
config also keeps the deprecated `stochastic`/`init_noise_std`/
`noise_std_type`/`state_dependent_std` fields populated; IsaacLab's
`handle_deprecated_rsl_rl_cfg(agent_cfg, version)` will strip them at
runtime (called from `scripts/train.py` and `scripts/play.py`).

---

## 11. Training scripts

- `scripts/train.py`: builds the env + `RslRlVecEnvWrapper` and runs
  `OnPolicyRunner.learn`. Avoids Hydra; lifts CLI overrides directly so
  the package stays vendor-friendly.
- `scripts/play.py`: loads a checkpoint and rolls out
  `--num_steps` of inference.
- `scripts/smoke_test.py`: zero-action rollout with shape assertions
  (`obs == (E, 393)`, `act == (E, 22)`); used as a CI sanity check.

All three call `AppLauncher` first, before any IsaacLab / Isaac Sim
import, which is mandatory for the Omniverse runtime.

---

## 12. Pitfalls encountered (and how we fixed them)

These caused real failures during the migration; recording them here so
future maintainers don't re-discover them:

1. **URDF mesh names with `.` in the stem confuse PXR's `SdfPath`.**
   Allegro's `link_0.0.obj` etc. caused
   `ValueError: Failed to convert MeshConfig`. Fix: rename
   `*.0.*` → `*_0.*` in `assets/robot/ur5_description/allegro_meshes/`
   and update the URDF references. Same for the long Robotiq mesh name
   (`robotiq_ft300-G-062-COUPLING_G-50-4M6-1D6_20181119.STL`) which we
   collapsed to `robotiq_ft300_coupling.STL`.
2. **`UrdfConverter` defaults `merge_fixed_joints=True`,** which collapses
   `right_gripper_palm_link` into its parent during USD conversion. The
   palm `ContactSensor` then fails with "no rigid bodies under prim".
   Fix: explicitly set `merge_fixed_joints=False` on the robot's
   `UrdfFileCfg`. The 4 finger parent links survive either way; the palm
   is the only one we need this for.
3. **`activate_contact_sensors` must be set on BOTH spawners.** Robot
   side enables the reporter API; object side makes the YCB body a valid
   contact partner. Setting only one side gives "could not find any
   bodies with contact reporter API" or empty `force_matrix_w` even when
   contacts visibly happen.
4. **`filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]`.** Using a global
   regex (`/World/envs/.*/Object`) returns one `force_matrix_w` per env
   that aggregates contacts from all envs and triggers the
   "expected N, found 2" assertion. The per-env regex resolves to
   exactly one filter body per env and gives a deterministic shape.
5. **`InteractiveSceneCfg.clone_in_fabric=True` breaks rendering for
   complex articulations.** PhysX still simulates correctly, but only
   `env_0` shows up visually. We default to `clone_in_fabric=False`.
6. **`ManagerBasedRLEnv` instantiation order.** Two consequences:
   - The object's `RigidObjectCfg` has to be on `cfg.scene` **before**
     `super().__init__`, hence the dynamic patching in
     `AllegroRelocateManagerEnv.__init__`.
   - The `ObsTerm` dry-run runs inside `super().__init__ → load_managers`
     and tickles every observation function with arbitrary state.
     `goal_state` reads `env._traj_*`, so we override `load_managers` to
     allocate those buffers first, then defer to the parent.
7. **`current_step` per env.** Reward/termination depend on it; we
   increment in our `step()` override (after PhysX advances, before the
   manager runs reward / termination terms) and reset in
   `reset_trajectory_state`.
8. **Trajectory object name.** The `.npz` files store
   `object_name="006_mustard_bottle"`. Older code stripped the leading
   `006_`, breaking the path lookup
   `assets/ycb/006_mustard_bottle/textured_simple.obj`. We keep the
   full name (matching the `vividex_sapien/assets/ycb_models` directory
   layout).
9. **rsl-rl-lib 5.0.0 dropped `policy=`.** Old IsaacLab examples still
   ship a `RslRlPpoActorCriticCfg`. Using it raises
   `KeyError: 'class_name'` deep inside
   `rsl_rl.algorithms.ppo.construct_algorithm`. The fix is to define
   `actor`/`critic` as separate `RslRlMLPModelCfg`s and call
   `handle_deprecated_rsl_rl_cfg` to strip the legacy MISSING fields.
10. **Block-wise flattening of robot link states.** Initially the 22-link
    block was flattened as `(pos, quat, lv, av)` per link, which gives
    393 dims but a different layout from ViViDex. We re-derived the
    layout from
    `vividex_sapien/.../base.py::get_oracle_state` and switched to
    block-wise concatenation (all 22 pos, then all 22 quat, …).
11. **Robot base pose must mirror `lab.ROBOT2BASE`.** In ViViDex the
    UR5 is *not* placed at the world origin: every reset calls
    `self.robot.set_pose(Pose(p=ROBOT2BASE.p + root_offset))` where
    `ROBOT2BASE.p = (0.765, -0.09, 0)`. The recorded trajectory's
    `object_translation` and `robot_jpos` are in the same world frame,
    so canonical `(x, y) = (0.35, 0.35)` is roughly `(-0.42, +0.44)` in
    the **arm-base** frame. Pinning the robot at `(0, 0, 0)` while
    keeping the object at `(0.35, 0.35)` puts the arm to the wrong
    side of the object (arm appears on the left of the object instead
    of behind / right). Fix: set
    `ArticulationCfg.init_state.pos = ROBOT_BASE_POS = (0.765, -0.09,
    TABLE_TOP_Z)` and re-centre the table at `(0.5625, 0)` so it
    spans both robot footprint and object area.
12. **PhysX `gpu_total_aggregate_pairs_capacity` must scale with
    `num_envs`.** At 4096 envs PhysX prints
    `The application needs to increase
    PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to ~70k`,
    after which broad-phase silently misses contacts. The cfg default
    in `__post_init__` is keyed off the cfg-time `num_envs`, but the
    user's CLI override happens *after* the cfg is constructed. We
    therefore recompute the capacity in
    `AllegroRelocateManagerEnv.__init__` (before `super().__init__`) as
    `max(16384, 64 × num_envs)`, which gives ~2× headroom over the
    empirical 17 pairs/env seen at 4096 envs.

---

## 13. External usage

To use this package from another project:

```bash
pip install -e /root/workspace/rl_grasp/isaaclab_dextrous_grasp
```

Then:

```python
import gymnasium as gym
import isaaclab_dextrous_grasp  # registers the gym ID

env = gym.make("Isaac-AllegroUR5-Relocate-v0", cfg=env_cfg)
```

The env follows the standard IsaacLab manager-based contract, so it
plugs into any third-party `RslRlVecEnvWrapper` /
`RslRlOnPolicyRunnerCfg` workflow.

---

## 14. Verification checklist

Run these commands in the `env_isaaclab` conda environment:

```bash
# 1. Sanity-check shapes and a 20-step zero-action rollout
python scripts/smoke_test.py --num_envs 4 --headless

# 2. End-to-end PPO learn iteration
python scripts/train.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 64 --headless --max_iterations 1
```

A successful pass prints `obs.shape == (4, 393)`, `act dims == 22`, and
the rsl-rl runner banner with `Mean action std: 0.20` for iteration 0.
