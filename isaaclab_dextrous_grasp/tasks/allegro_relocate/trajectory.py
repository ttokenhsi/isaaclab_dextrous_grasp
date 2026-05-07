"""Trajectory loading + per-env buffer management.

This module mirrors the trajectory pre-processing that
``vividex_sapien.hand_imitation.env.rl_env.relocate_env.AllegroRelocateRLEnv``
performs in its ``__init__`` and ``reset`` methods, but vectorised across all
parallel envs.

Key responsibilities:

1. Load a ``.npz`` trajectory (norm_traj format) and split it into a
   *pregrasp* segment and an *imitate* segment, mirroring ViViDex's
   ``robot_pregrasp_jpos`` / ``robot_jpos`` split at ``pregrasp_step``.
2. Pad / truncate the imitate segment to a constant length so all envs share
   the same ``imitate_steps`` / ``traj_len`` (matches ViViDex's
   ``constant_steps = 60`` for relocate).
3. Sample per-env curriculum parameters ``(x, y, theta_z)`` according to the
   ``stage`` (0/1/2) curriculum and apply them as canonicalisation → rotation
   → translation, exactly like ViViDex's ``reset``.
4. Store all per-env buffers as torch tensors on the simulation device, ready
   for indexing by the manager-based env's reward/observation/event terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .env_paths import trajectory_path


# Default constants - copied from ViViDex.
PREGRASP_STEPS: int = 15
CONSTANT_IMITATE_STEPS: int = 60  # SAPIEN: ``constant_steps`` for relocate
LIFT_KEYFRAME_HEIGHT: float = 0.1  # imitate up to lifting by 0.1 m


@dataclass
class TrajectoryStatic:
    """The trajectory representation *before* per-env randomisation.

    This stores a single canonical reference trajectory (object centred at
    origin, palm/fingers expressed relative to the table), padded to length
    ``imitate_steps`` exactly. Per-env randomisation is then applied at reset
    time to produce the per-env tensors.
    """

    object_name: str  # e.g. "006_mustard_bottle"
    init_object_height: float

    # Pregrasp segment: 5 joints (palm + 4 fingertips)
    # Shape (P+1, 5, 3) where P is the *original* pregrasp_step
    robot_pregrasp_jpos: np.ndarray

    # Imitate segment, all padded to imitate_steps:
    object_translation: np.ndarray  # (T, 3)
    object_orientation: np.ndarray  # (T, 4) wxyz
    robot_jpos: np.ndarray          # (T, 5, 3)

    # Misc
    pregrasp_steps: int  # number of control steps reserved for pregrasp
    imitate_steps: int   # = pregrasp_steps + (T-1) after padding


def load_trajectory(name: str) -> TrajectoryStatic:
    """Load and pre-process a ViViDex norm trajectory ``.npz`` file."""

    path = trajectory_path(name)
    raw = np.load(str(path))
    data = {k: raw[k] for k in raw.files}

    object_name = str(data["object_name"])
    # ViViDex strips the leading "00X_" prefix because their YCB asset
    # directory layout uses the short name; we keep the original full name
    # because our bundled assets/ycb/<full_name>/ folder retains the prefix.
    short_name = object_name

    init_object_height = float(data["init_object_height"])
    pregrasp_step = int(data["pregrasp_step"])

    # ----- split pregrasp / imitate exactly like ViViDex -----
    robot_jpos_full = data["robot_jpos"]                     # (T_full, 5, 3)
    robot_pregrasp_jpos = robot_jpos_full[: pregrasp_step + 1].copy()  # (P+1, 5, 3)
    robot_jpos = robot_jpos_full[pregrasp_step:].copy()                 # (T, 5, 3)
    object_translation = data["object_translation"][pregrasp_step:].copy()  # (T, 3)
    object_orientation = data["object_orientation"][pregrasp_step:].copy()  # (T, 4) wxyz

    # ----- canonicalise: subtract object[0], lift by init_object_height -----
    obj0 = object_translation[0].copy()
    object_translation = object_translation - obj0
    robot_jpos = robot_jpos - obj0
    robot_pregrasp_jpos = robot_pregrasp_jpos - obj0
    object_translation[:, 2] += init_object_height
    robot_jpos[:, :, 2] += init_object_height
    robot_pregrasp_jpos[:, :, 2] += init_object_height

    # NOTE: per-env rotation/translation is applied later at reset time.

    # ----- truncate at lift > 0.1 m, mirroring vividex non-norm path -----
    # In ViViDex's norm_traj path, this truncation is *not* applied.  We follow
    # the norm_traj behaviour (no lift truncation); the padding step below
    # extends the trajectory as needed.
    pregrasp_steps = PREGRASP_STEPS
    raw_imitate_len = pregrasp_steps + len(object_translation) - 1
    target = max(raw_imitate_len, CONSTANT_IMITATE_STEPS)

    # Pad by repeating the last frame, matching ViViDex.
    pad = target - raw_imitate_len
    if pad > 0:
        last_t = object_translation[-1:].repeat(pad, axis=0)
        last_q = object_orientation[-1:].repeat(pad, axis=0)
        last_j = robot_jpos[-1:].repeat(pad, axis=0)
        object_translation = np.concatenate([object_translation, last_t], axis=0)
        object_orientation = np.concatenate([object_orientation, last_q], axis=0)
        robot_jpos = np.concatenate([robot_jpos, last_j], axis=0)

    imitate_steps = target

    return TrajectoryStatic(
        object_name=short_name,
        init_object_height=init_object_height,
        robot_pregrasp_jpos=robot_pregrasp_jpos.astype(np.float32),
        object_translation=object_translation.astype(np.float32),
        object_orientation=object_orientation.astype(np.float32),
        robot_jpos=robot_jpos.astype(np.float32),
        pregrasp_steps=pregrasp_steps,
        imitate_steps=imitate_steps,
    )


# ---------------------------------------------------------------------------
#                               Per-env buffers
# ---------------------------------------------------------------------------


@dataclass
class PerEnvTrajectoryBuffers:
    """All per-env trajectory tensors live on the simulation device."""

    # (E, T, 3)
    obj_t: torch.Tensor
    # (E, T, 4) wxyz
    obj_q: torch.Tensor
    # (E, T, 5, 3) palm + 4 fingertips
    jpos: torch.Tensor
    # (E, P+1, 5, 3)
    pregrasp_jpos: torch.Tensor
    # (E, 3) per-env (x, y, init_object_height)
    init_pos: torch.Tensor


def allocate_buffers(static: TrajectoryStatic, num_envs: int, device: torch.device) -> PerEnvTrajectoryBuffers:
    """Pre-allocate per-env buffers on the given device.

    Filled with the *un-randomised* canonical trajectory broadcast across envs;
    actual randomisation happens in :func:`apply_stage_randomisation`.
    """

    obj_t = torch.tensor(static.object_translation, device=device).unsqueeze(0).expand(num_envs, -1, -1).clone()
    obj_q = torch.tensor(static.object_orientation, device=device).unsqueeze(0).expand(num_envs, -1, -1).clone()
    jpos = torch.tensor(static.robot_jpos, device=device).unsqueeze(0).expand(num_envs, -1, -1, -1).clone()
    pre = torch.tensor(static.robot_pregrasp_jpos, device=device).unsqueeze(0).expand(num_envs, -1, -1, -1).clone()
    init_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
    init_pos[:, 2] = static.init_object_height
    return PerEnvTrajectoryBuffers(
        obj_t=obj_t, obj_q=obj_q, jpos=jpos, pregrasp_jpos=pre, init_pos=init_pos
    )


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (... , 4) wxyz quaternions."""

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def _yaw_quat_wxyz(theta: torch.Tensor) -> torch.Tensor:
    """Build (..., 4) wxyz quaternions for a rotation about z by ``theta``."""

    half = theta * 0.5
    w = torch.cos(half)
    x = torch.zeros_like(theta)
    y = torch.zeros_like(theta)
    z = torch.sin(half)
    return torch.stack([w, x, y, z], dim=-1)


def _yaw_rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Build (E, 3, 3) yaw rotation matrices."""

    c = torch.cos(theta)
    s = torch.sin(theta)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    row0 = torch.stack([c, -s, zero], dim=-1)
    row1 = torch.stack([s, c, zero], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # (E, 3, 3)


def sample_stage_params(
    stage: int,
    env_ids: torch.Tensor,
    device: torch.device,
    init_object_height: float,
    stage3_xy_range: tuple[float, float] = (0.20, 0.40),
    stage3_yaw_abs: float = float(np.pi / 6.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample per-env ``(x, y, theta_z, init_z)`` for the given curriculum stage.

    Stage map::

        0  canonical (x, y, theta) = (0.35, 0.35, 0)
        1  x, y ∼ U[0.30, 0.40],            theta = 0
        2  x, y ∼ U[0.30, 0.40],            theta ∼ U[-pi/12, +pi/12]
        3+ x, y ∼ U[stage3_xy_range],       theta ∼ U[-stage3_yaw_abs,
                                                       +stage3_yaw_abs]

    Stages 0/1/2 mirror ViViDex's curriculum exactly. Stage 3 is our
    extension for harder generalisation: defaults to ±10 cm xy + ±30° yaw
    (2× wider than stages 1/2 in both dimensions). Both ranges are sourced
    from ``TaskCfg.stage3_xy_range`` / ``stage3_yaw_abs`` via
    :func:`apply_stage_randomisation` so they can be tuned without editing
    this module.

    Returns four (N,) float32 tensors aligned to ``env_ids`` order.
    """

    n = env_ids.shape[0]
    if stage <= 0:
        x = torch.full((n,), 0.35, device=device, dtype=torch.float32)
        y = torch.full((n,), 0.35, device=device, dtype=torch.float32)
        theta = torch.zeros((n,), device=device, dtype=torch.float32)
    elif stage == 1:
        x = torch.empty((n,), device=device, dtype=torch.float32).uniform_(0.30, 0.40)
        y = torch.empty((n,), device=device, dtype=torch.float32).uniform_(0.30, 0.40)
        theta = torch.zeros((n,), device=device, dtype=torch.float32)
    elif stage == 2:
        x = torch.empty((n,), device=device, dtype=torch.float32).uniform_(0.30, 0.40)
        y = torch.empty((n,), device=device, dtype=torch.float32).uniform_(0.30, 0.40)
        theta = torch.empty((n,), device=device, dtype=torch.float32).uniform_(
            -np.pi / 12.0, np.pi / 12.0
        )
    else:
        # stage >= 3: wider xy + wider yaw (configurable from TaskCfg).
        x_lo, x_hi = float(stage3_xy_range[0]), float(stage3_xy_range[1])
        x = torch.empty((n,), device=device, dtype=torch.float32).uniform_(x_lo, x_hi)
        y = torch.empty((n,), device=device, dtype=torch.float32).uniform_(x_lo, x_hi)
        yaw_abs = float(stage3_yaw_abs)
        theta = torch.empty((n,), device=device, dtype=torch.float32).uniform_(
            -yaw_abs, yaw_abs
        )
    init_z = torch.full((n,), float(init_object_height), device=device, dtype=torch.float32)
    return x, y, theta, init_z


def apply_stage_randomisation(
    buffers: PerEnvTrajectoryBuffers,
    static: TrajectoryStatic,
    env_ids: torch.Tensor,
    stage: int,
    stage3_xy_range: tuple[float, float] = (0.20, 0.40),
    stage3_yaw_abs: float = float(np.pi / 6.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-fill per-env trajectory buffers with newly sampled (x, y, theta_z).

    Returns ``(x, y, theta_z)`` so the caller can reuse them when placing the
    object / robot.

    The optional ``stage3_xy_range`` / ``stage3_yaw_abs`` knobs only kick in
    for ``stage >= 3``; stages 0/1/2 keep ViViDex's hard-coded ranges. They
    are forwarded straight to :func:`sample_stage_params`, normally fed by
    ``mdp/events.reset_trajectory_state`` from ``env.cfg.task``.

    Implementation notes
    --------------------
    Mirrors ViViDex's ``reset``:

    1. Start from canonical (centred) tensors stored on the static trajectory
       (object[0] = (0, 0, init_z), all 5-joint refs already lifted).
    2. Build per-env yaw rotation R(theta), apply to ``object_translation``,
       ``robot_jpos`` and ``robot_pregrasp_jpos``.
    3. Multiply ``object_orientation`` by yaw quaternion (Hamilton product
       order: ``q_new = R_yaw_quat * q_old``, matching ViViDex's
       ``Quaternion(matrix=rot_matrix @ Q.rotation_matrix)``).
    4. Translate (xy only) by per-env (x, y).
    """

    device = buffers.obj_t.device
    x, y, theta, init_z = sample_stage_params(
        stage,
        env_ids,
        device,
        static.init_object_height,
        stage3_xy_range=stage3_xy_range,
        stage3_yaw_abs=stage3_yaw_abs,
    )
    n = env_ids.shape[0]

    # Canonical references (broadcast on demand to (n, ...)) ----------------
    obj_t_canon = torch.tensor(static.object_translation, device=device).unsqueeze(0).expand(n, -1, -1)
    obj_q_canon = torch.tensor(static.object_orientation, device=device).unsqueeze(0).expand(n, -1, -1)
    jpos_canon = torch.tensor(static.robot_jpos, device=device).unsqueeze(0).expand(n, -1, -1, -1)
    pre_canon = torch.tensor(static.robot_pregrasp_jpos, device=device).unsqueeze(0).expand(n, -1, -1, -1)

    # Rotation matrices (n, 3, 3) and yaw quats (n, 4) ----------------------
    R = _yaw_rotation_matrix(theta)
    yaw_q = _yaw_quat_wxyz(theta)

    # Apply rotations -------------------------------------------------------
    # obj_t: (n, T, 3) - rotate around z
    obj_t = torch.einsum("nij,ntj->nti", R, obj_t_canon)
    # jpos: (n, T, 5, 3)
    jpos = torch.einsum("nij,ntkj->ntki", R, jpos_canon)
    pre = torch.einsum("nij,nkqj->nkqi", R, pre_canon)
    # obj_q (n, T, 4): yaw_q * obj_q_canon (broadcast yaw across time)
    yaw_q_t = yaw_q.unsqueeze(1).expand(-1, obj_q_canon.shape[1], -1)
    obj_q = _quat_mul_wxyz(yaw_q_t, obj_q_canon)

    # Apply translation in x, y --------------------------------------------
    trans = torch.zeros((n, 3), device=device, dtype=torch.float32)
    trans[:, 0] = x
    trans[:, 1] = y
    obj_t = obj_t + trans.unsqueeze(1)
    jpos = jpos + trans.unsqueeze(1).unsqueeze(1)
    pre = pre + trans.unsqueeze(1).unsqueeze(1)

    # Write into the per-env buffers ---------------------------------------
    buffers.obj_t[env_ids] = obj_t
    buffers.obj_q[env_ids] = obj_q
    buffers.jpos[env_ids] = jpos
    buffers.pregrasp_jpos[env_ids] = pre
    init_pos = torch.zeros((n, 3), device=device, dtype=torch.float32)
    init_pos[:, 0] = x
    init_pos[:, 1] = y
    init_pos[:, 2] = init_z
    buffers.init_pos[env_ids] = init_pos

    return x, y, theta
