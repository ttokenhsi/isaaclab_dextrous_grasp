"""IK correctness tests for the AllegroUR5 relocate env.

Four progressive tests, each isolating one potential failure mode:

A. **Single-step IK with small target** -- DLS damping shrinks per-step
   Δq, so a 50 mm target won't be reached in one step (this is by design,
   matches vividex). Use a 5 mm target where damping bias is negligible.

A'. **Iterated IK** -- repeatedly call IK with the same large target and
    write Δq into joint state. Should converge to the target in ~10 iterations
    with sub-mm residual.

B. **vividex parity** -- numpy DLS vs IsaacLab DifferentialIK on the same
   Jacobian + same v_des. Should agree to ~1e-6.

C. **IK + PD tracking** -- like A but uses ``set_joint_position_target`` and
   runs physics for one control step. Compares the residual against Test A
   to isolate PD-side error (e.g. missing velocity feedforward).

C'. **IK + PD with velocity feedforward** -- same as C but also calls
    ``set_joint_velocity_target(arm_qvel)``. If C >> C', the missing
    feedforward is the culprit.

Usage::

    python scripts/test_ik.py --num_envs 4 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--small_offset", type=float, default=0.005,
               help="Small target pose offset for test A (m).")
p.add_argument("--big_offset", type=float, default=0.05,
               help="Larger target pose offset for tests A'/C/C' (m).")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym

import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

import isaaclab_dextrous_grasp  # noqa: F401
from isaaclab_dextrous_grasp.tasks.allegro_relocate.manager_env_cfg import (
    AllegroRelocateManagerEnvCfg,
)


def quat_angle(q: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.arccos(q[..., 0].clamp(-1.0, 1.0).abs())


def quat_mul_inverse(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return math_utils.quat_mul(q1, math_utils.quat_inv(q2))


def vividex_dls_ik(twist_world: np.ndarray, J: np.ndarray, damping: float = 0.05) -> np.ndarray:
    """Verbatim reproduction of vividex's `compute_inverse_kinematics`."""
    lmbda = np.eye(6) * (damping ** 2)
    return J.T @ np.linalg.lstsq(J @ J.T + lmbda, twist_world, rcond=None)[0]


def reset_arm(robot, arm_joint_ids, arm_q0):
    jp = robot.data.joint_pos.clone()
    jp[:, arm_joint_ids] = arm_q0
    jv = torch.zeros_like(robot.data.joint_vel)
    robot.write_joint_state_to_sim(jp, jv)


def get_jacobian(robot, jacobi_body_idx, jacobi_arm_joint_ids):
    return robot.root_physx_view.get_jacobians()[
        :, jacobi_body_idx, :, jacobi_arm_joint_ids
    ]


# ---------------------------------------------------------------------------
cfg = AllegroRelocateManagerEnvCfg()
cfg.scene.num_envs = args.num_envs
env = gym.make("Isaac-AllegroUR5-Relocate-v0", cfg=cfg)
unw = env.unwrapped
unw.reset()

robot = unw.scene["robot"]
device = unw.device
n = unw.num_envs

action_term = unw.action_manager.get_term("arm_hand")
arm_joint_ids = action_term._arm_joint_ids
palm_idx = action_term._palm_body_idx
jacobi_body_idx = action_term._jacobi_body_idx
jacobi_arm_joint_ids = action_term._jacobi_arm_joint_ids
ik_damping = action_term.cfg.ik_damping

palm_pos_w0 = robot.data.body_pos_w[:, palm_idx].clone()
palm_quat_w0 = robot.data.body_quat_w[:, palm_idx].clone()
arm_q0 = robot.data.joint_pos[:, arm_joint_ids].clone()

env_origins = unw.scene.env_origins

ik_cfg = DifferentialIKControllerCfg(
    command_type="pose", use_relative_mode=False, ik_method="dls",
    ik_params={"lambda_val": ik_damping},
)


def run_ik_one_step(target_pos, target_quat, palm_pos, palm_quat, arm_q):
    """Run one IK step and return the desired arm qpos."""
    ik = DifferentialIKController(ik_cfg, num_envs=n, device=device)
    ik.set_command(
        torch.cat([target_pos, target_quat], dim=-1), palm_pos, palm_quat,
    )
    J = get_jacobian(robot, jacobi_body_idx, jacobi_arm_joint_ids)
    return ik.compute(palm_pos, palm_quat, J, arm_q)


def write_arm_qpos_and_refresh(arm_qpos):
    jp = robot.data.joint_pos.clone()
    jp[:, arm_joint_ids] = arm_qpos
    jv = torch.zeros_like(robot.data.joint_vel)
    robot.write_joint_state_to_sim(jp, jv)
    unw.sim.forward()
    unw.scene.update(dt=0.0)


def palm_state():
    return (
        robot.data.body_pos_w[:, palm_idx].clone(),
        robot.data.body_quat_w[:, palm_idx].clone(),
    )


# ===========================================================================
# TEST A -- Pure IK math, SMALL target (damping bias negligible)
# ===========================================================================
print("\n" + "#" * 78)
print(f"# TEST A  -- Single-step IK, small target ({args.small_offset*1000:.0f} mm)")
print("#" * 78)
reset_arm(robot, arm_joint_ids, arm_q0); unw.sim.forward(); unw.scene.update(dt=0.0)
palm_pos, palm_quat = palm_state()
target_pos = palm_pos + torch.tensor(
    [args.small_offset, 0, 0], device=device
).unsqueeze(0).expand(n, -1)
target_quat = palm_quat
arm_qpos_des = run_ik_one_step(target_pos, target_quat, palm_pos, palm_quat, arm_q0)
write_arm_qpos_and_refresh(arm_qpos_des)
palm_pos_now, palm_quat_now = palm_state()
err_a = (palm_pos_now - target_pos).norm(dim=-1)
ach_a = (palm_pos_now - palm_pos).norm(dim=-1) / args.small_offset
print(f"target offset:                       {args.small_offset*1000:.1f} mm")
print(f"achieved offset / target  per-env:   {ach_a.cpu().numpy()}")
print(f"residual error per-env (mm):         {(err_a*1000).cpu().numpy()}")
test_a = bool((err_a < 1e-3).all())
print(f"TEST A  →  {'PASS' if test_a else 'FAIL'} (tol 1 mm)")


# ===========================================================================
# TEST A' -- Iterated IK, BIG target. Should converge.
# ===========================================================================
print("\n" + "#" * 78)
print(f"# TEST A' -- Iterated IK, big target ({args.big_offset*1000:.0f} mm)")
print("#" * 78)
reset_arm(robot, arm_joint_ids, arm_q0); unw.sim.forward(); unw.scene.update(dt=0.0)
palm_pos0, palm_quat0 = palm_state()
target_pos = palm_pos0 + torch.tensor(
    [args.big_offset, 0, 0], device=device
).unsqueeze(0).expand(n, -1)
target_quat = palm_quat0

print("iter | residual (mm)")
print("-----|---------------")
for it in range(15):
    palm_pos, palm_quat = palm_state()
    arm_q = robot.data.joint_pos[:, arm_joint_ids]
    arm_qpos_des = run_ik_one_step(target_pos, target_quat, palm_pos, palm_quat, arm_q)
    write_arm_qpos_and_refresh(arm_qpos_des)
    palm_pos_now, _ = palm_state()
    err = (palm_pos_now - target_pos).norm(dim=-1)
    print(f"  {it:2d} | {(err*1000).cpu().numpy()}")
    if (err < 1e-4).all():
        break
test_ap = bool((err < 1e-3).all())
print(f"TEST A' →  {'PASS' if test_ap else 'FAIL'} (tol 1 mm)")


# ===========================================================================
# TEST B -- vividex parity
# ===========================================================================
print("\n" + "#" * 78)
print("# TEST B  -- vividex parity (numpy DLS vs IsaacLab DifferentialIK)")
print("#" * 78)
reset_arm(robot, arm_joint_ids, arm_q0); unw.sim.forward(); unw.scene.update(dt=0.0)
palm_pos, palm_quat = palm_state()
dt_ctrl = float(unw.step_dt)
v_des_world = torch.tensor(
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device
).unsqueeze(0).expand(n, -1)

target_pos_b = palm_pos + v_des_world[:, :3] * dt_ctrl
arm_qpos_des_b = run_ik_one_step(target_pos_b, palm_quat, palm_pos, palm_quat, arm_q0)
delta_q_isaaclab = (arm_qpos_des_b - arm_q0).cpu().numpy()

J = get_jacobian(robot, jacobi_body_idx, jacobi_arm_joint_ids).cpu().numpy()
v_np = v_des_world.cpu().numpy()
delta_q_vividex = np.stack([
    vividex_dls_ik(v_np[i], J[i], damping=ik_damping) * dt_ctrl for i in range(n)
])
abs_err = np.abs(delta_q_isaaclab - delta_q_vividex)
print(f"Δq IsaacLab (env 0): {delta_q_isaaclab[0]}")
print(f"Δq vividex  (env 0): {delta_q_vividex[0]}")
print(f"max |IsaacLab - vividex| = {abs_err.max():.3e}")
test_b = abs_err.max() < 1e-5
print(f"TEST B  →  {'PASS' if test_b else 'FAIL (>1e-5)'}")


# ===========================================================================
# TEST C -- IK + PD WITHOUT velocity feedforward
# ===========================================================================
print("\n" + "#" * 78)
print(f"# TEST C  -- IK + PD, NO velocity feedforward (big target)")
print("#" * 78)
reset_arm(robot, arm_joint_ids, arm_q0); unw.sim.forward(); unw.scene.update(dt=0.0)
palm_pos, palm_quat = palm_state()
target_pos_c = palm_pos + torch.tensor(
    [args.big_offset, 0, 0], device=device
).unsqueeze(0).expand(n, -1)
target_quat_c = palm_quat
arm_qpos_des_c = run_ik_one_step(target_pos_c, target_quat_c, palm_pos, palm_quat, arm_q0)

# Run physics for 1 control step (decimation x sim_dt) chasing the target
target_full = robot.data.joint_pos.clone()
target_full[:, arm_joint_ids] = arm_qpos_des_c
for _ in range(unw.cfg.decimation):
    robot.set_joint_position_target(target_full)
    robot.write_data_to_sim()
    unw.sim.step(render=False)
    unw.scene.update(dt=unw.physics_dt)

palm_pos_c, _ = palm_state()
err_c = (palm_pos_c - target_pos_c).norm(dim=-1)
ach_c = (palm_pos_c - palm_pos).norm(dim=-1) / args.big_offset
print(f"achieved/target per-env: {ach_c.cpu().numpy()}")
print(f"residual error  per-env (mm): {(err_c*1000).cpu().numpy()}")
test_c = bool((err_c < 5e-3).all())
print(f"TEST C  →  {'PASS' if test_c else 'FAIL (>5 mm)'}")


# ===========================================================================
# TEST C' -- IK + PD WITH velocity feedforward
# ===========================================================================
print("\n" + "#" * 78)
print(f"# TEST C' -- IK + PD, WITH velocity feedforward (vividex-style)")
print("#" * 78)
reset_arm(robot, arm_joint_ids, arm_q0); unw.sim.forward(); unw.scene.update(dt=0.0)
palm_pos, palm_quat = palm_state()
target_pos_cp = palm_pos + torch.tensor(
    [args.big_offset, 0, 0], device=device
).unsqueeze(0).expand(n, -1)
target_quat_cp = palm_quat
arm_qpos_des_cp = run_ik_one_step(target_pos_cp, target_quat_cp, palm_pos, palm_quat, arm_q0)

# vividex-style: arm_qvel = J⁺ v_des, set_joint_velocity_target(arm_qvel).
v_des = torch.zeros((n, 6), device=device)
v_des[:, 0] = args.big_offset / dt_ctrl  # treat the BIG offset as v*dt
J = get_jacobian(robot, jacobi_body_idx, jacobi_arm_joint_ids).cpu().numpy()
arm_qvel = np.stack([
    vividex_dls_ik(v_des.cpu().numpy()[i], J[i], damping=ik_damping) for i in range(n)
])
arm_qvel = torch.from_numpy(arm_qvel).to(device).float()

target_qpos_full = robot.data.joint_pos.clone()
target_qpos_full[:, arm_joint_ids] = arm_qpos_des_cp
target_qvel_full = torch.zeros_like(robot.data.joint_vel)
target_qvel_full[:, arm_joint_ids] = arm_qvel

for _ in range(unw.cfg.decimation):
    robot.set_joint_position_target(target_qpos_full)
    robot.set_joint_velocity_target(target_qvel_full)
    robot.write_data_to_sim()
    unw.sim.step(render=False)
    unw.scene.update(dt=unw.physics_dt)

palm_pos_cp, _ = palm_state()
err_cp = (palm_pos_cp - target_pos_cp).norm(dim=-1)
ach_cp = (palm_pos_cp - palm_pos).norm(dim=-1) / args.big_offset
print(f"achieved/target per-env: {ach_cp.cpu().numpy()}")
print(f"residual error  per-env (mm): {(err_cp*1000).cpu().numpy()}")
test_cp = bool((err_cp < 5e-3).all())
print(f"TEST C' →  {'PASS' if test_cp else 'FAIL (>5 mm)'}")


# ===========================================================================
print("\n" + "=" * 78)
print(f"SUMMARY")
print("=" * 78)
print(f"A  (single-step IK, small target)  : {'PASS' if test_a  else 'FAIL'}")
print(f"A' (iterated IK,    big target)    : {'PASS' if test_ap else 'FAIL'}")
print(f"B  (vividex parity)                : {'PASS' if test_b  else 'FAIL'}")
print(f"C  (IK+PD, no  vel feedforward)    : {'PASS' if test_c  else 'FAIL'}")
print(f"C' (IK+PD, with vel feedforward)   : {'PASS' if test_cp else 'FAIL'}")
print("=" * 78)

env.close()
app.close()
