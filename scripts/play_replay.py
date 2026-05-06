"""Open-loop replay of a saved ViViDex action sequence on the *unmodified*
AllegroUR5 YCB relocate env.

The point of this script is to answer **a single question**: does the original
``isaaclab_dextrous_grasp`` env produce a successful grasp when fed the exact
22-d action sequence that ViViDex's trained policy produced in SAPIEN? If yes,
the env's physics / observations / IK are functionally fine and any closed-loop
gap is purely the policy reacting to slightly-different obs at t>=1. If no,
there is a real env-side bug we need to chase.

Usage::

    python scripts/play_replay.py \\
        --task Isaac-AllegroUR5-Relocate-v0 \\
        --num_envs 1 --stage 0 \\
        --replay_actions /tmp/replay/actions_<traj>_stage0.npz \\
        --video --video_length 60

The npz layout we expect (produced by ``vividex_sapien/tools/record_replay_data.py``)::

    actions      : (T, 22)   float32, in [-1, 1]
    observations : (T, 393)  float32, optional -- only used for diagnostic A/B
    object_pos   : (T, 3)    float32
    object_lift  : (T,)      float32
    rewards      : (T,)      float32
    initial_object_pos / initial_object_quat / initial_robot_qpos : optional
    init_x, init_y, init_object_height, pregrasp_steps, imitate_steps : scalars

Nothing else in the repo is touched. We simply build the env with its defaults,
``env.reset()`` once, and feed ``actions[t]`` into ``env.step()`` for T steps.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Open-loop ViViDex action replay on the stock IsaacLab env."
)
parser.add_argument("--task", type=str, default="Isaac-AllegroUR5-Relocate-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--trajectory", type=str, default=None,
                    help="Override TaskCfg.trajectory_name. If unset, the env's "
                         "default trajectory is used (must match the npz).")
parser.add_argument("--replay_actions", type=str, required=True,
                    help="Path to actions_<traj>_stage0.npz produced by "
                         "vividex_sapien/tools/record_replay_data.py.")
parser.add_argument("--num_steps", type=int, default=None,
                    help="Number of env.step() calls. Default = T from npz.")
parser.add_argument("--no_terminations", action="store_true",
                    help="Disable the early-termination terms "
                         "(pregrasp_failure / object_too_far / lost_contact) "
                         "so the replay always runs to T steps even if the "
                         "stock env would otherwise reset early.")
parser.add_argument("--video", action="store_true",
                    help="Record an mp4 of env 0 to "
                         "logs/<traj_obj>/<pose>/play_videos/.")
parser.add_argument("--video_length", type=int, default=60)
parser.add_argument("--video_dir", type=str, default=None,
                    help="Override the output dir for --video.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# CRITICAL: Video recording needs the Replicator / Camera pipeline to be
# loaded *at AppLauncher boot time*. If we don't flip this flag, the kit
# experience launches without the camera extensions, and the first call to
# ``env.render()`` (triggered by ``gym.wrappers.RecordVideo`` after step 0)
# falls into a slow lazy-init path that takes 60-90 s to bring up Replicator
# in the background. With the flag set everything is ready before reset, and
# the per-frame render cost drops to a few ms.
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- imports that touch IsaacLab must come *after* AppLauncher -------------
import gymnasium as gym
import numpy as np
import torch

import isaaclab_dextrous_grasp  # noqa: F401  (registers the gym task ID)
from isaaclab_dextrous_grasp.tasks.allegro_relocate.manager_env_cfg import (
    AllegroRelocateManagerEnvCfg,
)


def _strip_early_terminations(env_cfg: AllegroRelocateManagerEnvCfg) -> None:
    """Replace early-termination DoneTerms with stubs so only ``time_out`` fires.

    We do this by wrapping the term ``func`` to return all-False, which keeps
    the manager schema intact (no need to re-validate the cfg).
    """

    from isaaclab.managers import TerminationTermCfg as DoneTerm

    def _never(env, **_kwargs):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for name in ("pregrasp_failure", "object_too_far", "lost_contact"):
        term = getattr(env_cfg.terminations, name, None)
        if term is None:
            continue
        # Replace in-place with a fresh DoneTerm referencing the stub.
        setattr(
            env_cfg.terminations, name,
            DoneTerm(func=_never, params={}, time_out=False),
        )


def main() -> None:
    # 1. Build env from defaults; only flip the few knobs we explicitly want
    #    to touch. The whole point of this script is to NOT modify the env.
    env_cfg = AllegroRelocateManagerEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.task.stage = args_cli.stage
    if args_cli.trajectory is not None:
        env_cfg.task.trajectory_name = args_cli.trajectory

    if args_cli.no_terminations:
        _strip_early_terminations(env_cfg)
        # Make sure time_out can still fire and bound episode length.
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, 6.0)

    # 2. Load the saved action sequence.
    if not os.path.isfile(args_cli.replay_actions):
        raise FileNotFoundError(args_cli.replay_actions)
    npz = np.load(args_cli.replay_actions)
    actions_np = npz["actions"]                 # (T, 22) float32 in [-1, 1]
    T_npz = int(actions_np.shape[0])
    n_steps = args_cli.num_steps if args_cli.num_steps is not None else T_npz
    print(
        f"[INFO] Loaded {T_npz} actions from {args_cli.replay_actions}; "
        f"will replay {n_steps} steps across {args_cli.num_envs} env(s)."
    )
    if "init_x" in npz.files:
        print(
            f"[INFO] npz init: object pos≈({float(npz['init_x']):.3f},"
            f"{float(npz['init_y']):.3f},{float(npz['init_object_height']):.3f})  "
            f"pregrasp_steps={int(npz['pregrasp_steps'])}  "
            f"imitate_steps={int(npz['imitate_steps'])}"
        )

    # Recording camera close to env 0's tabletop so the bottle and hand fill
    # the frame. Same view as ``play_vividex.py`` and the SAPIEN-side
    # ``record_replay_data.py`` so the two videos can be diffed side-by-side.
    if args_cli.video:
        env_cfg.viewer.eye = (1.05, 0.95, 0.55)
        env_cfg.viewer.lookat = (0.40, 0.40, 0.18)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.resolution = (640, 480)

    # 3. Build the env. Only request rgb_array rendering when actually
    #    recording, otherwise gym still calls render() each step which slows
    #    things down for nothing.
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        if args_cli.video_dir is not None:
            video_dir = args_cli.video_dir
        else:
            traj = env_cfg.task.trajectory_name
            obj = traj.split("-")[1] if "-" in traj else "unknown"
            # ycb-006_mustard_bottle-... -> mustard_bottle
            obj = obj.split("_", 1)[-1] if "_" in obj else obj
            video_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs", obj, f"pose{args_cli.stage + 1}", "play_videos",
            )
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to: {video_dir}")

    device = env.unwrapped.device
    actions = torch.from_numpy(actions_np).to(device)

    # 4. Reset, then step through the replay.
    obs_dict, _ = env.reset()
    print(f"[INFO] env reset complete; obs['policy'].shape = "
          f"{tuple(obs_dict['policy'].shape)}")

    cum_reward = torch.zeros(args_cli.num_envs, device=device)
    max_lift = torch.zeros(args_cli.num_envs, device=device)
    final_lift = torch.zeros(args_cli.num_envs, device=device)
    finished = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)

    base_env = env.unwrapped
    while hasattr(base_env, "env"):
        base_env = base_env.env

    obj_asset = base_env.scene["object"]
    init_object_z = obj_asset.data.root_pos_w[:, 2].clone()

    for t in range(n_steps):
        if t >= actions.shape[0]:
            print(f"[INFO] action sequence exhausted at t={t}; stopping.")
            break
        a = actions[t].unsqueeze(0).expand(args_cli.num_envs, -1).clone()
        obs_dict, reward, terminated, truncated, info = env.step(a)
        cum_reward = torch.where(finished, cum_reward, cum_reward + reward)
        cur_lift = obj_asset.data.root_pos_w[:, 2] - init_object_z
        max_lift = torch.where(finished, max_lift, torch.maximum(max_lift, cur_lift))
        final_lift = torch.where(finished, final_lift, cur_lift)
        # Bookkeeping: once an env is done, freeze its stats.
        finished = finished | (terminated | truncated)
        if t in (0, 9, 14, 19, 29, 44, 59) or t == n_steps - 1:
            print(
                f"[step {t:3d}]  reward(env0)={float(reward[0]):+.4f}  "
                f"cum(env0)={float(cum_reward[0]):+.4f}  "
                f"lift(env0)={float(cur_lift[0])*1000:+6.1f} mm  "
                f"max_lift(env0)={float(max_lift[0])*1000:+6.1f} mm  "
                f"done(env0)={bool(finished[0])}"
            )

    # 5. Final summary.
    print()
    print("=" * 72)
    print(f"[REPLAY DONE]  envs={args_cli.num_envs}  steps={t + 1}")
    print(f"  cum reward   : mean={cum_reward.mean().item():+.4f}  "
          f"min={cum_reward.min().item():+.4f}  "
          f"max={cum_reward.max().item():+.4f}")
    print(f"  max lift (mm): mean={max_lift.mean().item()*1000:+.1f}  "
          f"max={max_lift.max().item()*1000:+.1f}")
    print(f"  final lift   : mean={final_lift.mean().item()*1000:+.1f} mm  "
          f">20mm in {(final_lift > 0.02).sum().item()}/{args_cli.num_envs} envs")
    print("=" * 72)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
