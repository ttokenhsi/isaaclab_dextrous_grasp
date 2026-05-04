# Isaac Lab Dextrous Grasp

Standalone IsaacLab port of the
[ViViDex (SAPIEN)](https://github.com/zerchen/vividex_sapien) dextrous grasp
benchmark. A UR5 arm equipped with a Wonik Allegro hand learns to relocate YCB
objects, guided by reference motion trajectories recovered from human videos.

The package is **standalone**: it does not depend on the original
`vividex_sapien` repository at runtime. All necessary assets (UR5 + Allegro
URDF + meshes, YCB OBJ files, sample trajectories) are bundled under
`assets/` and `trajectories/`.

## Installation

This project targets the IsaacLab Python environment that already has
Isaac Sim, IsaacLab and `rsl-rl-lib` installed:

```bash
conda activate env_isaaclab
pip install -e /root/workspace/rl_grasp/isaaclab_dextrous_grasp
```

> The first time you launch the environment, the URDF and YCB meshes are
> converted to USD and cached under `cache/usd/`. This can take a minute per
> object; subsequent runs are instant.

## Usage

A single gym ID is registered using IsaacLab's manager-based RL framework:

| Gym ID                            | Framework                  | Entry point                                |
| --------------------------------- | -------------------------- | ------------------------------------------ |
| `Isaac-AllegroUR5-Relocate-v0`    | `ManagerBasedRLEnv`        | `manager_env.AllegroRelocateManagerEnv`    |

### Train

```bash
cd /root/workspace/rl_grasp/isaaclab_dextrous_grasp
python scripts/train.py --task Isaac-AllegroUR5-Relocate-v0 --num_envs 4096 --headless
```

### Train with video recording

`--video` periodically records `env_0` to
`logs/rsl_rl/<experiment>/<run>/videos/train/`.
`--video_length` (env steps) sets each clip's duration and
`--video_interval` (env steps) controls how often a new clip starts.
Cameras are enabled automatically when `--video` is set.

```bash
python scripts/train.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 4096 --headless \
    --video --video_length 200 --video_interval 2000
```

### Play / evaluate

```bash
python scripts/play.py \
    --task Isaac-AllegroUR5-Relocate-v0 \
    --num_envs 4 \
    --checkpoint logs/rsl_rl/allegro_ur5_relocate/<run>/model_<iter>.pt
```

### Smoke test

```bash
python scripts/smoke_test.py --num_envs 4
```

## Task specification

| Aspect | Value | Notes |
| --- | --- | --- |
| Robot  | UR5 + Allegro Hand (22 DoF) | URDF in `assets/robot/ur5_description` |
| Object | YCB rigid body                | OBJ files in `assets/ycb`, converted to USD on first use |
| Action space | 22-dim, ∈ `[-1, 1]`     | `[0:3]` palm linear vel, `[3:6]` palm angular vel, `[6:22]` Allegro qpos targets |
| Observation  | 393-dim oracle state    | `robot_state(330) + object_state(13) + goal_state(42) + time_state(8)` |
| Reward       | tracking + contact + lift bonus − controller/action penalty | matches ViViDex `get_reward` |
| Done         | pregrasp-fail / lost-contact / out-of-range / timeout | mirrors SAPIEN `is_done` |
| Decimation   | 10 (~50 Hz control)     | matches SAPIEN `frame_skip=10` |
| Contact      | IsaacLab `ContactSensor` | 5 sensors (palm + 4 finger parents), filtered to per-env Object |

### Trajectory format

The `.npz` files under `trajectories/` are the same as those produced by the
`vividex_sapien/norm_trajectories` directory:

```
robot_qpos          (T_q, 22)        UR5 + Allegro joint trajectory
robot_jpos          (T,   5, 3)      palm + 4 fingertip positions (world)
object_translation  (T,   3)
object_orientation  (T,   4)         quaternion (w, x, y, z)
init_object_height  ()
pregrasp_step       ()
SIM_SUBSTEPS, length, ...
```

Drop new trajectories in `trajectories/` and reference them via
`AllegroRelocateManagerEnvCfg.task.trajectory_names`.

### Curriculum (`task.stage`)

| Stage | Behaviour                                                                 |
| ----- | ------------------------------------------------------------------------- |
| 0     | Trajectory placed at canonical `(x=0.35, y=0.35)` with no rotation        |
| 1     | Per-env `(x, y) ∼ U([0.30, 0.40] × [0.30, 0.40])`                         |
| 2     | Stage-1 randomisation + per-env `θ_z ∼ U(-π/12, +π/12)`                   |

## Layout

```
isaaclab_dextrous_grasp/
├── assets/
│   ├── robot/ur5_description/
│   └── ycb/
├── trajectories/
├── cache/usd/
├── docs/migration.md
├── scripts/
│   ├── train.py
│   ├── play.py
│   └── smoke_test.py
└── isaaclab_dextrous_grasp/
    ├── __init__.py
    └── tasks/
        └── allegro_relocate/
            ├── __init__.py
            ├── env_paths.py
            ├── trajectory.py
            ├── manager_env_cfg.py
            ├── manager_env.py
            ├── mdp/
            │   ├── actions.py
            │   ├── observations.py
            │   ├── rewards.py
            │   ├── terminations.py
            │   └── events.py
            └── agents/
                └── rsl_rl_ppo_cfg.py
```

See [`docs/migration.md`](docs/migration.md) for the full
SAPIEN ↔ IsaacLab alignment / migration notes.
