"""Termination functions corresponding to ViViDex's :py:meth:`is_done`.

Mapping::

    pregrasp_failure         : at the pregrasp boundary, fingertip error > 5cm
    object_too_far           : in imitate phase, ||obj - traj_target|| > threshold (default 0.15)
    lost_contact_in_imitate  : in imitate phase, no contacting finger
    time_out (truncation)    : current_step >= imitate_steps
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import rewards as _rewards

if TYPE_CHECKING:
    from ..manager_env import AllegroRelocateManagerEnv


def pregrasp_failure(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    cache = _rewards._ensure_intermediates(env)
    # Trigger only at the boundary (current_step == pregrasp_steps), if pregrasp not yet flagged success.
    boundary = env.current_step >= env._pregrasp_steps
    not_yet_grasped = ~env._pregrasp_success
    failed = cache["pre_err"] > 0.05
    return boundary & not_yet_grasped & failed


def object_too_far(env: "AllegroRelocateManagerEnv", threshold: float = 0.15) -> torch.Tensor:
    cache = _rewards._ensure_intermediates(env)
    in_imitate = ~cache["in_pregrasp"]
    too_far = cache["obj_com_err"] > threshold
    return in_imitate & too_far


def lost_contact_in_imitate(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    cache = _rewards._ensure_intermediates(env)
    in_imitate = ~cache["in_pregrasp"]
    no_contact = cache["has_contact"] < 0.5
    return in_imitate & no_contact


def time_out(env: "AllegroRelocateManagerEnv") -> torch.Tensor:
    return env.current_step >= env._imitate_steps
