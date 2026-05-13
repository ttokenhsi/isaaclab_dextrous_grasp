"""Load a ViViDex (Stable-Baselines3 PPO) checkpoint and play it on the
IsaacLab AllegroUR5 YCB relocate task.

Both sides agree on:
  * obs: 393-d oracle state in the order
        [robot_state(330), object_state(13), goal_state(42), time_state(8)]
  * action: 22-d in [-1, 1]
        action[:6]  -> palm spatial velocity (3 lin + 3 ang)
        action[6:]  -> hand qpos targets (linearly mapped to joint limits)
  * MLP: Linear(393,256) -> Tanh -> Linear(256,128) -> Tanh -> Linear(128,22)

The ViViDex SB3 zip stores ``policy.pth`` whose keys we map verbatim into a
plain :class:`torch.nn.Module`. We do not depend on stable-baselines3 itself.

Usage::

    conda activate env_isaaclab
    python scripts/play_vividex.py \\
        --task Isaac-AllegroUR5-Relocate-v0 \\
        --num_envs 4 --stage 0 \\
        --checkpoint logs/mustard_bottle/pose1/restore_checkpoint.zip
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play a ViViDex SB3 checkpoint inside IsaacLab."
)
parser.add_argument("--task", type=str, default="Isaac-AllegroUR5-Relocate-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to ViViDex restore_checkpoint.zip (SB3 PPO format).",
)
parser.add_argument(
    "--stage",
    type=int,
    default=0,
    help="Curriculum stage (0=canonical (0.35,0.35,0deg) only). Match the stage "
    "the ckpt was trained on; pose1 was norm_traj=True stage 0/1/2 mix but "
    "stage 0 is the simplest deterministic eval setting.",
)
parser.add_argument(
    "--trajectory",
    type=str,
    default=None,
    help="Override env_cfg.task.trajectory_name. Defaults to mustard bottle pose1 traj.",
)
parser.add_argument("--num_steps", type=int, default=400)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record a video of env 0 to <ckpt_dir>/play_videos/.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length (env steps) of each recorded video segment.",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=0,
    help=(
        "Number of env steps between recording triggers. 0 = only record once "
        "starting at step 0 (single segment of --video_length steps)."
    ),
)
parser.add_argument(
    "--video_per_episode",
    action="store_true",
    default=False,
    help="Record one video per episode of env 0 (uses episode_trigger instead "
    "of step_trigger; ignores --video_interval).",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=True,
    help="Use the policy mean (no Gaussian sample). Default True.",
)
parser.add_argument(
    "--stochastic",
    dest="deterministic",
    action="store_false",
    help="Sample from the diagonal Gaussian using log_std from the ckpt.",
)
parser.add_argument(
    "--match_sapien_dt",
    action="store_true",
    default=False,
    help=(
        "Force-override env timing to ``sim.dt=1/200, decimation=10`` (5 ms / "
        "50 ms control = 20 Hz), matching ViViDex/SAPIEN's "
        "``scene.set_timestep(0.005)`` × ``frame_skip=10``. This is also the "
        "IsaacLab cfg default (see ``manager_env_cfg.AllegroRelocateManagerEnvCfg.__post_init__``), "
        "so the flag is mostly for symmetry / explicitness. Default False (= "
        "rely on the cfg default)."
    ),
)
parser.add_argument(
    "--no_match_sapien_dt",
    dest="match_sapien_dt",
    action="store_false",
    help="Disable the explicit timing override (cfg default still applies).",
)
parser.add_argument(
    "--replay_actions",
    type=str,
    default=None,
    help=(
        "Path to an ``actions_<env>_stage0.npz`` file produced by "
        "``vividex_sapien/tools/record_replay_data.py``. When set, the policy "
        "checkpoint is still loaded for shape verification but the per-step "
        "action vectors are taken from the file instead of policy.act(). Use "
        "with --num_envs 1 --video for a side-by-side rollout comparison."
    ),
)
parser.add_argument(
    "--only_timeout",
    action="store_true",
    default=False,
    help=(
        "Disable pregrasp_failure / object_too_far / lost_contact terminations "
        "so the only thing that ends an episode is the time_out (== max "
        "imitate steps). Useful for visualising what the policy is trying to "
        "do across the whole horizon without early termination."
    ),
)
parser.add_argument(
    "--fixes",
    type=str,
    default="",
    help=(
        "Comma-separated grasp-stability tweaks to apply. Choices: "
        "vel_iter (robot+object solver_velocity_iteration_count=4), "
        "stab (PhysX enable_stabilization=True), "
        "depen (object max_depenetration_velocity=1.0), "
        "fric (friction_offset_threshold=0.02), "
        "all (= vel_iter,stab,depen,fric)."
    ),
)
parser.add_argument(
    "--quiet_trace",
    action="store_true",
    default=False,
    help="Skip the per-step trace output so the log is just episode summaries.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Video recording requires cameras to be enabled in the AppLauncher.
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that depend on the IsaacSim app being launched.
# ---------------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import isaaclab_dextrous_grasp  # noqa: F401  (registers the gym task ID)
from isaaclab_dextrous_grasp.tasks.allegro_relocate.manager_env_cfg import (
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    AllegroRelocateManagerEnvCfg,
)


def _resolve_sapien_qpos_idx(base_env, robot) -> torch.Tensor:
    """Return a (22,) long tensor that re-orders IsaacLab's
    ``joint_pos`` / ``joint_vel`` into SAPIEN/URDF declaration order
    ``[arm_0..arm_5, joint_00, joint_01, ..., joint_15]``.

    Prefers ``base_env._qpos_sapien_idx`` when the env exposes it (newer env
    code) and otherwise rebuilds the mapping locally from joint names so the
    script keeps working on branches that haven't merged that buffer yet.
    """

    cached = getattr(base_env, "_qpos_sapien_idx", None)
    if cached is not None:
        return cached
    sapien_order = list(ARM_JOINT_NAMES) + list(HAND_JOINT_NAMES)
    ids, _ = robot.find_joints(sapien_order, preserve_order=True)
    if len(ids) != 22:
        raise RuntimeError(
            f"Could not resolve all 22 robot joints in URDF order; got {ids}"
        )
    idx = torch.tensor(ids, device=base_env.device, dtype=torch.long)
    # Cache on the env so subsequent calls (e.g. by reward terms) reuse it.
    base_env._qpos_sapien_idx = idx
    return idx


# ---------------------------------------------------------------------------
# ViViDex SB3 policy reconstruction
# ---------------------------------------------------------------------------


class VividexMLPPolicy(nn.Module):
    """Reproduces SB3 ``ActorCriticPolicy`` (default Tanh, no obs normalization).

    Layout matches the keys saved in ``policy.pth``::

        mlp_extractor.policy_net.0  Linear(393, 256)
        mlp_extractor.policy_net.2  Linear(256, 128)
        action_net                  Linear(128, 22)
        log_std                     (22,)              # state-independent

    For IsaacLab inference we ignore the value head; only ``forward`` returning
    the (mean, log_std) is needed. We don't need ortho-init either since we are
    loading a trained state dict.
    """

    def __init__(
        self,
        obs_dim: int = 393,
        act_dim: int = 22,
        net_arch: tuple[int, int] = (256, 128),
    ):
        super().__init__()
        h1, h2 = net_arch
        # SB3 stores Linear at indices 0 and 2 because of the interleaved Tanh.
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(h2, act_dim)
        # SB3's DiagGaussianDistribution stores a state-independent log_std.
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.policy_net(obs)
        return self.action_net(h), self.log_std

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            eps = torch.randn_like(mean)
            action = mean + eps * std
        # SB3's ``predict()`` clamps the policy output to the action_space box
        # whenever ``squash_output=False`` (which is what this checkpoint uses).
        # See ``stable_baselines3.common.policies.BasePolicy.predict`` and
        # ``algos/policies.ActorCriticPolicy`` -- the trained policy regularly
        # produces means outside [-1, 1] (e.g. arm components at ~ -1.27); SB3
        # clips them, IsaacLab's env (``recover_action``-style mapping) also
        # assumes the input is in [-1, 1]. Without this clamp we silently
        # send out-of-range commands that look correct but make the controller
        # extrapolate well past the joint limits, which is the root cause of
        # "policy can't grasp while replayed actions can".
        return torch.clamp(action, -1.0, 1.0)


def _load_sb3_policy_state_dict(ckpt_path: str) -> dict:
    """Extract ``policy.pth`` from an SB3 zip and return the state dict.

    SB3 zips contain (at minimum)::
        data, policy.pth, policy.optimizer.pth, pytorch_variables.pth,
        _stable_baselines3_version
    """

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(ckpt_path, "r") as zf:
            zf.extract("policy.pth", td)
        # weights_only=False is required because SB3 stores some non-tensor
        # metadata in the same file. Trust source: this is our own ckpt.
        sd = torch.load(
            os.path.join(td, "policy.pth"), map_location="cpu", weights_only=False
        )
    return sd


def _remap_sb3_keys_to_local(sd: dict) -> dict:
    """SB3 keys -> our :class:`VividexMLPPolicy` keys.

    SB3 ActorCriticPolicy:
        log_std
        mlp_extractor.policy_net.0.{weight,bias}
        mlp_extractor.policy_net.2.{weight,bias}
        mlp_extractor.value_net.0.{weight,bias}     <- ignored
        mlp_extractor.value_net.2.{weight,bias}     <- ignored
        action_net.{weight,bias}
        value_net.{weight,bias}                     <- ignored

    We only keep the policy / action / log_std weights.
    """

    out: dict = {}
    out["log_std"] = sd["log_std"]
    out["policy_net.0.weight"] = sd["mlp_extractor.policy_net.0.weight"]
    out["policy_net.0.bias"] = sd["mlp_extractor.policy_net.0.bias"]
    out["policy_net.2.weight"] = sd["mlp_extractor.policy_net.2.weight"]
    out["policy_net.2.bias"] = sd["mlp_extractor.policy_net.2.bias"]
    out["action_net.weight"] = sd["action_net.weight"]
    out["action_net.bias"] = sd["action_net.bias"]
    return out


def build_policy(ckpt_path: str, device: torch.device) -> VividexMLPPolicy:
    raw_sd = _load_sb3_policy_state_dict(ckpt_path)
    # Verify shapes so we fail loudly if the ckpt was trained with a different
    # arch (e.g. a vision policy with a different obs/action dim).
    obs_dim = raw_sd["mlp_extractor.policy_net.0.weight"].shape[1]
    h1 = raw_sd["mlp_extractor.policy_net.0.weight"].shape[0]
    h2 = raw_sd["mlp_extractor.policy_net.2.weight"].shape[0]
    act_dim = raw_sd["action_net.weight"].shape[0]
    print(
        f"[INFO] ViViDex ckpt: obs_dim={obs_dim} hidden=({h1},{h2}) act_dim={act_dim}"
    )
    if (obs_dim, act_dim) != (393, 22):
        raise RuntimeError(
            f"ckpt has obs_dim={obs_dim}, act_dim={act_dim}; "
            f"expected (393, 22) for the AllegroUR5 oracle policy."
        )

    policy = VividexMLPPolicy(obs_dim=obs_dim, act_dim=act_dim, net_arch=(h1, h2))
    local_sd = _remap_sb3_keys_to_local(raw_sd)
    missing, unexpected = policy.load_state_dict(local_sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    policy.to(device).eval()
    return policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    env_cfg = AllegroRelocateManagerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.task.stage = args_cli.stage
    if args_cli.trajectory is not None:
        env_cfg.task.trajectory_name = args_cli.trajectory

    # Align timing with ViViDex/SAPIEN training. SAPIEN config:
    #   ``relocate_env.py:26`` sets ``scene.set_timestep(0.005)`` (5 ms physics)
    #   ``relocate_env.py:36`` defaults ``frame_skip=10``           (10 sub-steps)
    #   => control_time_step = 0.005 * 10 = 0.05 s = 50 ms (20 Hz control)
    # IsaacLab's cfg default (sim.dt=1/200=5 ms, decimation=10 -> 50 ms) already
    # matches; this flag just re-asserts it for explicitness.
    if args_cli.match_sapien_dt:
        env_cfg.sim.dt = 1.0 / 200.0
        env_cfg.decimation = 10
        env_cfg.sim.render_interval = env_cfg.decimation
    print(
        f"[INFO] timing: sim.dt={env_cfg.sim.dt*1e3:.2f} ms, "
        f"decimation={env_cfg.decimation} -> control dt="
        f"{env_cfg.sim.dt*env_cfg.decimation*1e3:.2f} ms (vividex expects 50 ms)"
    )

    # Disable all early terminations -- only time_out remains. IsaacLab's
    # TerminationManager treats ``term_cfg is None`` as "skip this term".
    if args_cli.only_timeout:
        env_cfg.terminations.pregrasp_failure = None
        env_cfg.terminations.object_too_far = None
        env_cfg.terminations.lost_contact = None
        # Make sure episode_length_s is long enough that time_out fires only at
        # the end of the imitate horizon and not earlier.
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, 6.0)
        print(
            "[INFO] Early terminations disabled (pregrasp_failure / object_too_far "
            "/ lost_contact); only time_out remains. "
            f"episode_length_s = {env_cfg.episode_length_s}"
        )

    # ------------------------------------------------------------------
    # Optional grasp-stability tweaks (see manager_env_cfg.py defaults).
    # ------------------------------------------------------------------
    requested = {f.strip() for f in args_cli.fixes.split(",") if f.strip()}
    if "all" in requested:
        requested |= {"vel_iter", "stab", "depen", "fric"}
    applied: list[str] = []
    if "vel_iter" in requested:
        env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 12
        # The object cfg is built lazily in AllegroRelocateManagerEnv.__init__
        # so we stash the request on the env_cfg and the env's helper applies
        # it when constructing the object spawn.
        env_cfg.task.__dict__["_object_solver_iters"] = (12, 4)
        applied.append("vel_iter(4)")
    if "stab" in requested:
        env_cfg.sim.physx.enable_stabilization = True
        applied.append("stabilization")
    if "depen" in requested:
        env_cfg.task.__dict__["_object_max_depen_vel"] = 1.0
        applied.append("max_depen_vel(1.0)")
    if "fric" in requested:
        env_cfg.sim.physx.friction_offset_threshold = 0.02
        applied.append("friction_offset_thresh(0.02)")
    if applied:
        print(f"[INFO] Grasp-stability tweaks applied: {', '.join(applied)}")
    else:
        print("[INFO] No grasp-stability tweaks (baseline).")

    # Recording camera close to env 0's tabletop. The view target is the
    # canonical object spawn (env-local (0.40, 0.40, 0.18)) and the camera
    # sits diagonally in front of the robot. Mirrors the camera placement
    # used by ``vividex_sapien/tools/record_replay_data.py`` so the two
    # rollouts can be diffed side-by-side at the same viewing angle.
    if args_cli.video:
        env_cfg.viewer.eye = (1.05, 0.95, 0.55)
        env_cfg.viewer.lookat = (0.40, 0.40, 0.18)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.resolution = (640, 480)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        video_dir = os.path.join(os.path.dirname(ckpt_path), "play_videos")
        if os.path.isdir(video_dir):
            # Wipe so successive runs don't collide on the same step indices.
            shutil.rmtree(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        record_kwargs = {
            "video_folder": video_dir,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        if args_cli.video_per_episode:
            record_kwargs["episode_trigger"] = lambda ep: True
        elif args_cli.video_interval > 0:
            iv = args_cli.video_interval
            record_kwargs["step_trigger"] = lambda step: step % iv == 0
        else:
            record_kwargs["step_trigger"] = lambda step: step == 0
        env = gym.wrappers.RecordVideo(env, **record_kwargs)
        print(f"[INFO] Recording video to: {video_dir}")
        if args_cli.video_per_episode:
            print("[INFO]   trigger = every episode of env 0")
        elif args_cli.video_interval > 0:
            print(f"[INFO]   trigger = every {args_cli.video_interval} env steps")
        else:
            print("[INFO]   trigger = only at step 0 (single segment)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(ckpt_path, device)

    # Optional: load a saved (action_t) sequence from a ViViDex rollout to
    # replay open-loop here. This is the cleanest way to A/B the physics:
    # given identical actions and identical initial state, any divergence
    # between the bottle's trajectories is purely sim-vs-sim.
    replay_actions = None
    replay_obs_ref = None
    if args_cli.replay_actions is not None:
        data = np.load(args_cli.replay_actions)
        replay_actions = torch.from_numpy(data["actions"]).to(device)
        if "observations" in data.files:
            replay_obs_ref = torch.from_numpy(data["observations"]).to(device)
        print(
            f"[INFO] Replay mode: loaded {replay_actions.shape[0]} actions from "
            f"{args_cli.replay_actions}; the policy will NOT be evaluated."
        )

        # --------------------------------------------------------------
        # Sanity-check the rebuilt policy: feed it the *exact* obs that
        # SAPIEN saw at every step and compare its (deterministic) action
        # to the action that ViViDex's policy actually produced. Same
        # checkpoint + same obs => actions must match to ~1e-6.
        # --------------------------------------------------------------
        if replay_obs_ref is not None:
            with torch.no_grad():
                # build_policy returned the same VividexMLPPolicy used at runtime.
                pred_act_full = policy.act(
                    replay_obs_ref.to(policy.action_net.weight.dtype),
                    deterministic=True,
                )
            ref_act_full = replay_actions  # (T, 22)
            T = min(pred_act_full.shape[0], ref_act_full.shape[0])
            pred_act_full = pred_act_full[:T]
            ref_act_full = ref_act_full[:T]
            d = (pred_act_full - ref_act_full).cpu().numpy()
            ref_np_full = ref_act_full.cpu().numpy()
            pred_np = pred_act_full.cpu().numpy()
            print()
            print(
                "[POLICY-A/B] feed npz obs[t] -> rebuilt policy.act(deterministic) "
                f"vs npz actions[t]   T={T}"
            )
            print("  step    max|d|     L2(d)     ||ref||   rel%")
            for t in [0, 1, 2, 5, 10, 20, 30, 40, 50, T - 1]:
                if t >= T or t < 0:
                    continue
                dt = d[t]
                refn = float(np.linalg.norm(ref_np_full[t]))
                dl2 = float(np.linalg.norm(dt))
                rel = (dl2 / refn * 100.0) if refn > 1e-8 else 0.0
                print(
                    f"  {t:4d}  {np.abs(dt).max():.4e}  {dl2:.4e}  "
                    f"{refn:.4e}  {rel:5.2f}%"
                )
            print()
        if args_cli.num_envs != 1:
            print(
                f"[WARN] --num_envs={args_cli.num_envs} but replay actions are "
                "single-env; the same action vector will be applied to every env."
            )

    # ManagerBasedRLEnv returns (obs_dict, info). The "policy" obs group is
    # the 393-d concatenated tensor we built in observations.py.
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"].to(device)
    assert obs.shape[-1] == 393, f"unexpected obs dim {obs.shape}"

    # Per-segment diff of env 0 step-0 obs vs the saved ViViDex obs.
    if replay_obs_ref is not None:
        ref0 = replay_obs_ref[0]                # (393,)
        cur0 = obs[0]                           # (393,)
        diff = (cur0 - ref0).cpu().numpy()
        ref_np = ref0.cpu().numpy()
        cur_np = cur0.cpu().numpy()
        # Layout (see observations.py docstring + ViViDex relocate_env.py):
        #   [  0: 22) qpos        URDF order
        #   [ 22: 44) qvel
        #   [ 44:110) joint_link_pos    22 bodies × 3
        #   [110:198) joint_link_quat   22 bodies × 4
        #   [198:264) joint_link_linvel 22 × 3
        #   [264:330) joint_link_angvel 22 × 3
        #   [330:333) object_pos
        #   [333:337) object_quat
        #   [337:340) object_lin_vel
        #   [340:343) object_ang_vel
        #   [343:364) goal: 3 future frames × (orn4+trans3) = 21
        #   [364:367) palm-obj diff
        #   [367:379) 4 fingertip-obj diffs
        #   [379:382) palm-target diff
        #   [382:385) obj-target diff
        #   [385:393) time_state
        segments = [
            ("qpos",                   0,  22),
            ("qvel",                  22,  44),
            ("link_pos (22*3)",       44, 110),
            ("link_quat (22*4)",     110, 198),
            ("link_linvel (22*3)",   198, 264),
            ("link_angvel (22*3)",   264, 330),
            ("object_pos",           330, 333),
            ("object_quat",          333, 337),
            ("object_lin_vel",       337, 340),
            ("object_ang_vel",       340, 343),
            ("traj_goals (21)",      343, 364),
            ("palm-obj diff",        364, 367),
            ("4 finger-obj diff",    367, 379),
            ("palm-target diff",     379, 382),
            ("obj-target diff",      382, 385),
            ("time_state (8)",       385, 393),
        ]
        print()
        print("[OBS-DIFF env0 step0] IsaacLab vs ViViDex (saved in npz)")
        print(f"  {'segment':22s}  {'max|d|':>10s}  {'mean|d|':>10s}  "
              f"{'L2(d)':>10s}  {'||ref||':>10s}  rel%")
        for name, lo, hi in segments:
            seg = diff[lo:hi]
            ref_seg = ref_np[lo:hi]
            ref_norm = float(np.linalg.norm(ref_seg))
            d_l2 = float(np.linalg.norm(seg))
            rel = (d_l2 / ref_norm * 100.0) if ref_norm > 1e-8 else 0.0
            print(
                f"  {name:22s}  {np.abs(seg).max():10.4e}  "
                f"{np.abs(seg).mean():10.4e}  {d_l2:10.4e}  "
                f"{ref_norm:10.4e}  {rel:5.2f}%"
            )
        # (verbose component-wise dump removed -- the segment table above is
        # enough to confirm bit-identical t=0 obs between IsaacLab and SAPIEN.)

    # ------------------------------------------------------------------
    # One-shot debug: object pose after reset + traj-buffer pregrasp target.
    # ------------------------------------------------------------------
    import math as _math
    base_env = env.unwrapped
    while hasattr(base_env, "env"):
        base_env = base_env.env
    scene = base_env.scene
    obj = scene["object"]
    robot = scene["robot"]
    env_origin = scene.env_origins[0]

    # ------- object pose (env-local) -------
    obj_pos_local = (obj.data.root_pos_w[0] - env_origin).cpu().numpy()
    obj_quat_w = obj.data.root_quat_w[0].cpu().numpy()  # wxyz, frame-independent
    buf_obj_q0 = base_env._traj_buffers.obj_q[0, 0].cpu().numpy()
    buf_obj_t0 = base_env._traj_buffers.obj_t[0, 0].cpu().numpy()
    pre_target = base_env._traj_pregrasp[0, -1, 1:].cpu().numpy()  # (4 fingertips, 3)

    # ------- robot base / palm / wrist3 / fingertip poses (env-local) -------
    base_link_w = robot.data.root_pos_w[0].cpu().numpy()
    base_pos_local = (robot.data.root_pos_w[0] - env_origin).cpu().numpy()
    base_quat_w = robot.data.root_quat_w[0].cpu().numpy()
    palm_pos_local = (
        robot.data.body_pos_w[0, base_env._palm_body_idx] - env_origin
    ).cpu().numpy()
    palm_quat_w = robot.data.body_quat_w[0, base_env._palm_body_idx].cpu().numpy()
    finger_pos_local = (
        robot.data.body_pos_w[0, base_env._finger_body_idx] - env_origin.unsqueeze(0)
    ).cpu().numpy()

    # wrist_3_link (end of arm chain, before the gripper) for reference.
    wrist3_ids, _ = robot.find_bodies(["right_wrist_3_link"], preserve_order=True)
    wrist3_pos_local = (
        robot.data.body_pos_w[0, wrist3_ids[0]] - env_origin
    ).cpu().numpy()

    # ------- robot joint qpos in SAPIEN URDF order -------
    # Compute the URDF -> IsaacLab joint index map locally so the script
    # works whether or not the env exposes ``_qpos_sapien_idx`` itself.
    sapien_idx = _resolve_sapien_qpos_idx(base_env, robot)
    qpos_urdf = robot.data.joint_pos[0, sapien_idx].cpu().numpy()  # (22,)
    arm_qpos = qpos_urdf[:6]
    hand_qpos = qpos_urdf[6:22]

    print("[DEBUG] env0 ----- initial state after reset -----")
    print("  Robot base (URDF root):")
    print(f"    pos in env-local frame : {base_pos_local}")
    print(f"    pos in world frame     : {base_link_w}     "
          f"(env_origin={env_origin.cpu().numpy()})")
    print(f"    quat (wxyz)            : {base_quat_w}")
    print("  Arm joint qpos (URDF order, 6 UR5 joints):")
    arm_names = [
        "shoulder_pan", "shoulder_lift", "elbow",
        "wrist_1", "wrist_2", "wrist_3",
    ]
    for nm, q in zip(arm_names, arm_qpos):
        print(f"    {nm:14s} = {q:+.4f} rad ({q*180.0/_math.pi:+7.2f} deg)")
    print("  Hand joint qpos (URDF order, joint_00..joint_15):")
    for k, q in enumerate(hand_qpos):
        print(f"    joint_{k:02d}  = {q:+.4f} rad")
    print("  Arm-end / hand link cartesian poses (env-local):")
    print(f"    wrist_3_link  pos = {wrist3_pos_local}")
    print(f"    palm_link     pos = {palm_pos_local}    quat (wxyz) = {palm_quat_w}")
    for i, name in enumerate(["thumb", "index", "middle", "ring"]):
        print(f"    {name:6s} tip   pos = {finger_pos_local[i]}")
    print("  Object spawn pose (env-local):")
    print(f"    pos                = {obj_pos_local}")
    print(f"    quat (wxyz)        = {obj_quat_w}")
    print(f"    traj_buffers.obj_t[0,0] = {buf_obj_t0}     (expect spawn pos)")
    print(f"    traj_buffers.obj_q[0,0] = {buf_obj_q0}     (expect spawn quat)")
    print("  Pregrasp target fingertips (env-local):")
    for i, name in enumerate(["thumb", "index", "middle", "ring"]):
        d = float(((pre_target[i] - palm_pos_local) ** 2).sum() ** 0.5)
        print(f"    {name:6s} -> {pre_target[i]}    d_to_palm={d:.3f} m")
    pre_err0 = float(
        torch.linalg.norm(
            torch.from_numpy(finger_pos_local - pre_target), dim=-1
        ).mean()
    )
    print(f"  pre_err (mean over 4 fingertips at t=0): {pre_err0:.3f} m")
    # ---- Object & robot physics summary (mass / friction / torsional) ----
    try:
        obj_asset = env.unwrapped.scene["object"]
        obj_mass = float(obj_asset.root_physx_view.get_masses()[0].item())
        rmat = obj_asset.root_physx_view.get_material_properties()[0].cpu().numpy()
        print(
            f"  Object mass = {obj_mass:.3f} kg, friction (static/dyn/rest) = "
            f"({rmat[0,0]:.2f}/{rmat[0,1]:.2f}/{rmat[0,2]:.2f})"
        )
    except Exception as exc:
        print(f"  [could not query object material]: {exc!r}")
    try:
        rb = env.unwrapped.scene["robot"]
        sapien_idx = _resolve_sapien_qpos_idx(base_env, rb).cpu().numpy()
        eff = rb.root_physx_view.get_dof_max_forces()[0].cpu().numpy()
        # Print just one arm joint and one hand joint for sanity.
        joint_names = rb.joint_names
        arm_eff = eff[sapien_idx[0]]
        hand_eff = eff[sapien_idx[6]]
        print(
            f"  Joint effort_max (PhysX): arm={arm_eff:.1f} N·m, hand={hand_eff:.1f} N·m"
        )
    except Exception as exc:
        print(f"  [could not query joint effort]: {exc!r}")
    print("[DEBUG] -------------------------------------------")


    n_done_total = 0
    rewards_running = torch.zeros(args_cli.num_envs, device=device)
    steps_running = torch.zeros(args_cli.num_envs, device=device, dtype=torch.long)
    rewards_per_episode: list[float] = []
    lengths_per_episode: list[int] = []

    # Per-component cumulative reward over env 0 of episode 0 only (to compare
    # to ViViDex's per-term rewards).
    reward_mgr = base_env.reward_manager
    component_keys = list(reward_mgr._term_names)
    component_cum_e0: dict[str, float] = {k: 0.0 for k in component_keys}
    component_done_e0 = False

    # Trace the first ``trace_steps`` steps of env 0 in detail so we can
    # diagnose whether the policy / IK / PD is doing what we expect.
    trace_steps = 0 if args_cli.quiet_trace else 18  # > pregrasp_steps too
    # Imitate-phase per-step tracer for env 0 (always on - one episode worth
    # of lines is cheap and shows the full grasp / lift / drop story).
    imitate_trace_lines: list[str] = []

    for step in range(args_cli.num_steps):
        if replay_actions is not None:
            if step >= replay_actions.shape[0]:
                print(f"[INFO] replay sequence exhausted at step {step}; stopping.")
                break
            # Broadcast the (22,) replay action across the (num_envs, 22) batch
            # so that running with num_envs>1 simply runs num_envs identical
            # rollouts in parallel.
            action = replay_actions[step].unsqueeze(0).expand(args_cli.num_envs, -1).clone()
        else:
            action = policy.act(obs, deterministic=args_cli.deterministic)
        # IsaacLab env always clips internally, but we mirror SB3's predict()
        # which clips to action_space bounds before sending to the env.
        action = action.clamp(-1.0, 1.0)
        obs_dict, reward, terminated, truncated, _info = env.step(action)
        obs = obs_dict["policy"].to(device)

        if step < trace_steps:
            with torch.no_grad():
                a0 = action[0].cpu().numpy()
                cart_v = a0[:6]
                palm_now = (
                    scene["robot"].data.body_pos_w[0, base_env._palm_body_idx]
                    - env_origin
                ).cpu().numpy()
                cache = getattr(base_env, "_reward_cache", None)
                pre_err_val = float(cache["pre_err"][0]) if cache is not None else float("nan")
                cart_err_val = float(getattr(base_env, "_cartesian_error", torch.zeros(1))[0])
                in_pre = bool(base_env.current_step[0].item() <= base_env._pregrasp_steps[0].item())
                tag = "PRE " if in_pre else "IMIT"
                print(
                    f"[trace s{step:2d}|{tag}] "
                    f"a_cart=[{cart_v[0]:+.2f},{cart_v[1]:+.2f},{cart_v[2]:+.2f}|"
                    f"{cart_v[3]:+.2f},{cart_v[4]:+.2f},{cart_v[5]:+.2f}]  "
                    f"a_hand_max|min={a0[6:].max():+.2f}|{a0[6:].min():+.2f}  "
                    f"palm=[{palm_now[0]:.3f},{palm_now[1]:.3f},{palm_now[2]:.3f}]  "
                    f"pre_err={pre_err_val:.3f}  cart_err={cart_err_val:.4f}  "
                    f"r={float(reward[0]):+.3f}"
                )

        rewards_running += reward.to(device)
        steps_running += 1

        # Track per-component reward for env 0 first episode. The reward
        # manager stores ``value/dt`` in ``_step_reward``, so we multiply by
        # the env's step_dt to recover the true per-term contribution to the
        # episode return. (The reward functions already /10 internally; this
        # cumulative thus equals the ViViDex reward / 10.)
        if not component_done_e0:
            step_dt = float(base_env.step_dt)
            for key in component_keys:
                idx = reward_mgr._term_names.index(key)
                term_val = reward_mgr._step_reward[0, idx] * step_dt
                component_cum_e0[key] += float(term_val.item())
            from isaaclab_dextrous_grasp.tasks.allegro_relocate.mdp.rewards import (
                _ensure_intermediates as _ensure_inter,
            )
            cache = _ensure_inter(base_env)
            cb = cache["contact_bool"][0].cpu().numpy().astype(int)
            imitate_trace_lines.append(
                f"  s{step:2d}  pre={float(cache['pre_err'][0]):.3f}  "
                f"ftip={float(cache['fingertip_err'][0]):.3f}  "
                f"oc={float(cache['obj_com_err'][0]):.3f}  "
                f"orot={float(cache['obj_rot_err'][0]):.3f}  "
                f"lift={float(cache['obj_lift'][0]):.3f}  "
                f"contact[palm,t,i,m,r]={cb.tolist()}  "
                f"r={float(reward[0]):+.4f}"
            )
            base_env._reward_cache = None  # don't leak into next step

        done = terminated | truncated
        if done.any():
            for e in done.nonzero(as_tuple=False).flatten().tolist():
                rewards_per_episode.append(float(rewards_running[e].item()))
                lengths_per_episode.append(int(steps_running[e].item()))
                n_done_total += 1
                if e == 0 and not component_done_e0:
                    component_done_e0 = True
                    print("[IMITATE-TRACE env0 ep0]")
                    for line in imitate_trace_lines:
                        print(line)
                    print("[REWARD-BREAKDOWN env0 ep0] (already /10 like vividex)")
                    total = 0.0
                    for k in component_keys:
                        v = component_cum_e0[k]
                        total += v
                        print(f"   {k:24s} = {v:+.4f}")
                    print(f"   {'-- total --':24s} = {total:+.4f}")
                    print(
                        f"   ViViDex training ep_rew_mean (for ref): ~80.7 / "
                        f"60 steps = ~1.345 per step"
                    )
            rewards_running[done] = 0.0
            steps_running[done] = 0

        if (step + 1) % 50 == 0:
            avg = (
                sum(rewards_per_episode[-50:]) / max(1, len(rewards_per_episode[-50:]))
            )
            print(
                f"[step {step + 1:4d}] episodes_done={n_done_total} "
                f"recent_mean_return={avg:.3f}"
            )

    if rewards_per_episode:
        avg = sum(rewards_per_episode) / len(rewards_per_episode)
        avg_len = sum(lengths_per_episode) / len(lengths_per_episode)
        max_len = max(lengths_per_episode)
        print(
            f"[DONE] {len(rewards_per_episode)} episodes, "
            f"mean return = {avg:.3f}, mean length = {avg_len:.1f} (max {max_len})"
        )
    else:
        print(
            f"[DONE] no episode finished within {args_cli.num_steps} steps "
            f"(running mean return = {rewards_running.mean().item():.3f})"
        )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
