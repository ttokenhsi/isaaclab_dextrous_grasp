"""Gym registration for the AllegroUR5 YCB relocate task.

Importing this module registers the task ID
``Isaac-AllegroUR5-Relocate-v0`` with :mod:`gymnasium`. The registration uses
fully-qualified module paths so that ``isaaclab.envs.utils.spec_to_gym_tasks``
can later resolve the cfg / agent classes.
"""

import gymnasium as gym

from . import agents
from .manager_env import AllegroRelocateManagerEnv
from .manager_env_cfg import AllegroRelocateManagerEnvCfg

# ---------------------------------------------------------------------------
# Gym registration
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-AllegroUR5-Relocate-v0",
    entry_point=f"{AllegroRelocateManagerEnv.__module__}:{AllegroRelocateManagerEnv.__name__}",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{AllegroRelocateManagerEnvCfg.__module__}:{AllegroRelocateManagerEnvCfg.__name__}"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.rsl_rl_ppo_cfg.__name__}:AllegroUR5RelocatePPORunnerCfg"
        ),
    },
)

__all__ = ["AllegroRelocateManagerEnv", "AllegroRelocateManagerEnvCfg"]
