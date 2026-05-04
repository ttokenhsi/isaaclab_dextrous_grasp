"""rsl_rl PPO configuration aligned with vividex's ``ppo.yaml``.

This file targets ``rsl-rl-lib >= 5.0.0`` (which IsaacLab 5.x ships with). It
uses the new actor / critic ``RslRlMLPModelCfg`` API, ``obs_groups`` mapping,
and the legacy fields are NOT used.

ViViDex hyper-parameters (vividex_sapien/algos/rl/config/agent/ppo.yaml)::

    gamma: 0.95
    gae_lambda: 0.95
    learning_rate: 1e-5
    ent_coef: 0.001
    vf_coef: 0.5
    clip_range: 0.2
    n_steps: 4096          # global batch
    batch_size: 256        # SGD minibatch
    n_epochs: 5
    net_arch: pi=[256,128], vf=[256,128]
    log_std_init: -1.6     # init_std = exp(-1.6) ≈ 0.20

In rsl_rl's vectorised setting ``num_steps_per_env × num_envs ≈ 4096``, so we
pick ``num_steps_per_env=64`` to give 4096 with 64 envs (the default).
``num_mini_batches = num_steps_per_env * num_envs / batch_size = 4096 / 256 = 16``.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AllegroUR5RelocatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner cfg matching vividex's hyper-parameters."""

    seed: int = 42
    num_steps_per_env: int = 64
    """4096 / 64 envs = 64 steps each → matches vividex ``n_steps=4096``."""

    max_iterations: int = 5000
    save_interval: int = 50
    experiment_name: str = "allegro_ur5_relocate"

    # rsl_rl >= 5.0.0 mapping from observation groups to actor/critic obs sets.
    obs_groups: dict[str, list[str]] = {"actor": ["policy"], "critic": ["policy"]}

    # Deprecated convenience knob; kept for completeness with the base class.
    empirical_normalization: bool = False

    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.20,  # ≈ exp(-1.6)
            std_type="scalar",
        ),
        # Suppress deprecated MISSING fields.
        stochastic=True,
        init_noise_std=0.20,
        noise_std_type="scalar",
        state_dependent_std=False,
    )
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        stochastic=False,
        init_noise_std=0.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
