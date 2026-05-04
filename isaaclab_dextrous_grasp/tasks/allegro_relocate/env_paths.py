"""Asset / cache path helpers for the AllegroUR5 relocate task.

All paths are resolved relative to the *package root* (``isaaclab_dextrous_grasp``)
so the package is portable: as long as the package is installed (or its parent
directory is on ``sys.path``) the helpers work without any environment
variables.
"""

from __future__ import annotations

import os
from pathlib import Path


def _package_root() -> Path:
    """Return the absolute path to the project root that contains the
    ``assets/``, ``trajectories/`` and ``cache/`` directories.

    The Python sub-package lives at ``<root>/isaaclab_dextrous_grasp`` so we
    walk two ``parents`` up from this file (``.../tasks/allegro_relocate/env_paths.py``
    → ``.../tasks/allegro_relocate`` → ``.../tasks`` → ``.../isaaclab_dextrous_grasp``
    → ``<root>``).
    """

    return Path(__file__).resolve().parents[3]


PROJECT_ROOT: Path = _package_root()
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
ROBOT_DIR: Path = ASSETS_DIR / "robot" / "ur5_description"
ROBOT_URDF: Path = ROBOT_DIR / "ur5_allegro.urdf"
YCB_DIR: Path = ASSETS_DIR / "ycb"
TRAJECTORIES_DIR: Path = PROJECT_ROOT / "trajectories"
CACHE_DIR: Path = PROJECT_ROOT / "cache" / "usd"
ROBOT_USD_CACHE: Path = CACHE_DIR / "robot"
YCB_USD_CACHE: Path = CACHE_DIR / "ycb"


def ensure_cache_dirs() -> None:
    """Create the USD cache directories if they don't exist yet."""

    for d in (CACHE_DIR, ROBOT_USD_CACHE, YCB_USD_CACHE):
        os.makedirs(d, exist_ok=True)


def ycb_visual_obj(object_name: str) -> Path:
    """``object_name`` is e.g. ``006_mustard_bottle``."""

    return YCB_DIR / "visual" / object_name / "textured_simple.obj"


def ycb_collision_obj(object_name: str) -> Path:
    return YCB_DIR / "collision" / object_name / "collision.obj"


def ycb_usd_path(object_name: str) -> Path:
    """Where the USD converted from ``visual`` OBJ should live."""

    return YCB_USD_CACHE / object_name / "ycb.usd"


def trajectory_path(name: str) -> Path:
    """Resolve a trajectory ``name`` to an ``.npz`` file under
    ``trajectories/``. ``name`` can be either the basename without extension
    (``ycb-006_mustard_bottle-...``) or a full filename / absolute path.
    """

    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    if p.suffix == ".npz":
        candidate = TRAJECTORIES_DIR / p.name
    else:
        candidate = TRAJECTORIES_DIR / f"{name}.npz"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Trajectory '{name}' not found at '{candidate}'. "
            f"Available trajectories: {sorted(p.name for p in TRAJECTORIES_DIR.glob('*.npz'))}"
        )
    return candidate
