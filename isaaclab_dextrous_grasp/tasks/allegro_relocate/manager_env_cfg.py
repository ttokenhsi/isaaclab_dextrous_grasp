"""Manager-based env configuration for the AllegroUR5 YCB relocate task.

This file follows IsaacLab's recommended pattern (e.g.
``isaaclab_tasks.manager_based.manipulation.lift``): one configclass per MDP
manager (scene / observations / actions / rewards / terminations / events),
all wired together by :class:`AllegroRelocateManagerEnvCfg`.

The runtime env class :class:`AllegroRelocateManagerEnv` (defined in
``manager_env.py``) overrides :py:meth:`__init__` and :py:meth:`load_managers`
to:

* dynamically build the YCB ``RigidObjectCfg`` from the chosen trajectory's
  ``object_name`` (cannot be ``MISSING`` when validate runs),
* allocate per-env trajectory / state buffers **before** ObservationManager
  performs its dry-run on each ObsTerm.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UrdfFileCfg
from isaaclab.utils import configclass

from . import mdp
from .env_paths import ROBOT_URDF, ROBOT_USD_CACHE

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Shared constants (joint / link names, init pose, table geometry)
# ---------------------------------------------------------------------------

ARM_JOINT_NAMES: list[str] = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]
HAND_JOINT_NAMES: list[str] = [f"right_gripper_joint_{i:02d}" for i in range(16)]

# Joint links whose 13-d (pos+quat+linvel+angvel) state goes into ``robot_state``.
# Matches ViViDex's ``robot_joint_names`` exactly: 6 UR5 arm links + 16 Allegro
# hand links (4 fingers × 4 phalanges including base).
# With ``merge_fixed_joints=False`` all 22 of these are queryable PhysX bodies.
ROBOT_BODY_NAMES: list[str] = [
    "right_shoulder_link",
    "right_upper_arm_link",
    "right_forearm_link",
    "right_wrist_1_link",
    "right_wrist_2_link",
    "right_wrist_3_link",
    *[f"right_gripper_link_{i:02d}" for i in range(16)],
]
assert len(ROBOT_BODY_NAMES) == 22

PALM_BODY_NAME: str = "right_gripper_palm_link"
# vividex finger contact buckets (5): palm + thumb tip + index tip + middle tip + ring tip
# After merge_fixed_joints, the *_tip links collapse into their parent links.
FINGER_BODY_NAMES: list[str] = [
    "right_gripper_link_15",  # thumb tip parent
    "right_gripper_link_03",  # index tip parent
    "right_gripper_link_07",  # middle tip parent
    "right_gripper_link_11",  # ring tip parent
]


# UR5 init joint positions in degrees → radians.
ARM_INIT_QPOS: list[float] = [
    89.0 / 180.0 * math.pi,
    -116.0 / 180.0 * math.pi,
    147.0 / 180.0 * math.pi,
    -31.0 / 180.0 * math.pi,
    89.0 / 180.0 * math.pi,
    90.0 / 180.0 * math.pi,
]

# Hand init: all zeros except joint_12 (thumb base) = 0.5, matching vividex.
HAND_INIT_QPOS: dict[str, float] = {name: 0.0 for name in HAND_JOINT_NAMES}
HAND_INIT_QPOS["right_gripper_joint_12"] = 0.5

INIT_QPOS: dict[str, float] = {
    **{name: float(val) for name, val in zip(ARM_JOINT_NAMES, ARM_INIT_QPOS)},
    **HAND_INIT_QPOS,
}

# Table geometry. ViViDex uses a SAPIEN-side table at table_height=0.79; we
# match the top surface height so the YCB objects rest at z=0 + init_object_height.
TABLE_TOP_Z: float = 0.0  # we choose the world frame so that table top is at z=0
TABLE_HALF_SIZE: tuple[float, float, float] = (0.6, 0.6, 0.39)

# Robot base pose in the env-local frame. Mirrors vividex's
# ``lab.ROBOT2BASE = Pose(p=[0.765, -0.09, 0])`` so that the recorded
# trajectories (which encode object_translation / robot_jpos in the
# vividex world frame) line up correctly with the arm.
ROBOT_BASE_POS: tuple[float, float, float] = (0.765, -0.09, TABLE_TOP_Z)
# Centre the visual / collision table so it fully covers both the robot
# footprint at ROBOT_BASE_POS and the canonical object pose at (0.35, 0.35).
TABLE_CENTER_XY: tuple[float, float] = (0.5625, 0.0)


# ---------------------------------------------------------------------------
# Robot configuration helper
# ---------------------------------------------------------------------------


def _build_robot_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """Return an :class:`ArticulationCfg` for the UR5 + Allegro hand.

    ``activate_contact_sensors=True`` is required so PhysX exposes the contact
    reporter API on robot links (the 5 ContactSensors we add later need this).

    ``merge_fixed_joints=True`` (the IsaacLab default) collapses the
    ``*_tip`` fixed-joint sub-tree into the corresponding parent link. This is
    what allows the contact sensors at the parent links to actually receive
    contact data – with ``merge_fixed_joints=False`` PhysX refuses to
    initialise the reporter for the dangling fixed-joint sub-tree links.
    """

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=UrdfFileCfg(
            asset_path=str(ROBOT_URDF),
            usd_dir=str(ROBOT_USD_CACHE),
            usd_file_name="ur5_allegro.usd",
            fix_base=True,
            # NOTE: with merge_fixed_joints=True the URDF importer collapses
            # right_gripper_palm_link (and the entire FT300 / coupling chain)
            # into right_wrist_3_link, which makes it impossible to attach a
            # ContactSensor to the palm. We therefore keep fixed joints
            # un-merged so palm_link / *_tip links remain as queryable PhysX
            # rigid bodies (each carrying activate_contact_sensors=True).
            merge_fixed_joints=False,
            self_collision=False,
            activate_contact_sensors=True,
            joint_drive=UrdfFileCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=200.0, damping=10.0),
            ),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            articulation_props=schemas_cfg.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=INIT_QPOS,
            joint_vel={name: 0.0 for name in INIT_QPOS},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_NAMES,
                stiffness=20000.0,
                damping=400.0,
                effort_limit=300.0,
                velocity_limit=2.0 * math.pi,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=HAND_JOINT_NAMES,
                stiffness=200.0,
                damping=10.0,
                effort_limit=10.0,
                velocity_limit=2.0 * math.pi,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class AllegroRelocateSceneCfg(InteractiveSceneCfg):
    """Scene = ground + table + robot + YCB object + 5 contact sensors."""

    # filled in by :func:`_build_robot_cfg` from the env's ``__post_init__``.
    robot: ArticulationCfg = MISSING
    # filled by :class:`AllegroRelocateManagerEnv.__init__` once the trajectory
    # ``.npz`` is loaded (the YCB short name dictates which mesh to spawn).
    object: RigidObjectCfg = MISSING

    # ---- static scene props -------------------------------------------------

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(2.0 * TABLE_HALF_SIZE[0], 2.0 * TABLE_HALF_SIZE[1], 2.0 * TABLE_HALF_SIZE[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.85, 0.85),
                roughness=0.4,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.5, restitution=0.01
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_TOP_Z - TABLE_HALF_SIZE[2])
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.5)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # ---- contact sensors (palm + 4 finger groups) --------------------------
    palm_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{PALM_BODY_NAME}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    thumb_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{FINGER_BODY_NAMES[0]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    index_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{FINGER_BODY_NAMES[1]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    middle_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{FINGER_BODY_NAMES[2]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    ring_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{FINGER_BODY_NAMES[3]}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )


# ---------------------------------------------------------------------------
# Task / reward configclasses (mirror vividex `task_kwargs.reward_kwargs`)
# ---------------------------------------------------------------------------


@configclass
class TaskCfg:
    """Task knobs (control + curriculum + IK)."""

    trajectory_name: str = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"
    """``.npz`` filename (without extension) under ``trajectories/``."""

    stage: int = 0
    """Curriculum stage. 0 = canonical, 1 = (x,y) random, 2 = + theta_z random."""

    cart_lin_vel_limit: float = 1.0
    cart_ang_vel_limit: float = 1.0
    ik_damping: float = 0.05


@configclass
class RewardCfg:
    """Per-component scalings, all referenced from ``mdp/rewards.py``."""

    pregrasp_err_scale: float = 10.0
    object_track_err_scale: float = 50.0
    object_rot_err_weight: float = 0.1
    fingertip_track_err_scale: float = 10.0
    lift_bonus_thresh: float = 0.02
    obj_com_term: float = 0.15  # episode termination threshold for obj-target dist


# ---------------------------------------------------------------------------
# MDP manager configclasses
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    """22-dim mixed action: 6-d palm spatial vel via IK + 16-d hand qpos."""

    arm_hand: mdp.IKHandActionCfg = mdp.IKHandActionCfg(
        asset_name="robot",
        arm_joint_names=list(ARM_JOINT_NAMES),
        hand_joint_names=list(HAND_JOINT_NAMES),
        palm_body_name=PALM_BODY_NAME,
        cart_lin_vel_limit=1.0,
        cart_ang_vel_limit=1.0,
        ik_damping=0.05,
    )


@configclass
class ObservationsCfg:
    """Single 393-d ``policy`` group reproducing the ViViDex oracle state."""

    @configclass
    class PolicyGroup(ObsGroup):
        robot_state = ObsTerm(func=mdp.observations.robot_state)
        object_state = ObsTerm(func=mdp.observations.object_state)
        goal_state = ObsTerm(func=mdp.observations.goal_state)
        time_state = ObsTerm(func=mdp.observations.time_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyGroup = PolicyGroup()


@configclass
class RewardsCfg:
    """7 reward components, weights chosen so that the *sum* matches
    the ViViDex per-step reward divided by 10 (the ``/10`` factor is folded
    into the weights here).
    """

    pregrasp = RewTerm(func=mdp.rewards.pregrasp_reward, weight=1.0)
    contact = RewTerm(func=mdp.rewards.contact_reward, weight=1.0)
    object_track = RewTerm(func=mdp.rewards.object_track_reward, weight=1.0)
    fingertip_track = RewTerm(func=mdp.rewards.fingertip_track_reward, weight=1.0)
    lift_bonus = RewTerm(func=mdp.rewards.lift_bonus_reward, weight=1.0)
    controller_penalty = RewTerm(func=mdp.rewards.controller_penalty, weight=1.0)
    action_penalty = RewTerm(func=mdp.rewards.action_penalty, weight=1.0)


@configclass
class TerminationsCfg:
    """4 done conditions matching ViViDex ``is_done``."""

    pregrasp_failure = DoneTerm(func=mdp.terminations.pregrasp_failure)
    object_too_far = DoneTerm(
        func=mdp.terminations.object_too_far, params={"threshold": 0.15}
    )
    lost_contact = DoneTerm(func=mdp.terminations.lost_contact_in_imitate)
    time_out = DoneTerm(func=mdp.terminations.time_out, time_out=True)


@configclass
class EventsCfg:
    """Reset event = re-randomise per-env trajectory + reset robot/object state."""

    reset_traj = EventTerm(
        func=mdp.events.reset_trajectory_state,
        mode="reset",
        params={"stage": 0},
    )


# ---------------------------------------------------------------------------
# Top-level env cfg
# ---------------------------------------------------------------------------


@configclass
class AllegroRelocateManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL env config for AllegroUR5 YCB relocate."""

    # ---- Scene ------------------------------------------------------------
    scene: AllegroRelocateSceneCfg = AllegroRelocateSceneCfg(
        num_envs=64, env_spacing=2.5, replicate_physics=True
    )

    # ---- MDP managers -----------------------------------------------------
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # ---- Custom task knobs (not part of the IsaacLab schema) --------------
    task: TaskCfg = TaskCfg()
    reward: RewardCfg = RewardCfg()

    # ---- Decimation + sim --------------------------------------------------
    def __post_init__(self):
        # General settings: 50 Hz control = 200 Hz sim / 10 decimation.
        self.decimation = 10
        # An imitate trajectory is at most ~80 steps; at 50Hz that's 1.6s.
        # Give some slack for pregrasp + episode bookkeeping.
        self.episode_length_s = 4.0

        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # Total aggregate pairs capacity must scale with num_envs (PhysX warns
        # otherwise). Each env contributes ~17 broad-phase pairs (Robot ↔ Object
        # + ContactSensor filters), so we budget 32 / env with a generous floor.
        # 32 * 4096 = 131072 ⇒ headroom up to 8k envs at the default 32 ratio.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = max(
            16 * 1024, 32 * self.scene.num_envs
        )
        self.sim.physx.friction_correlation_distance = 0.00625

        # Build the robot cfg (object cfg is built lazily in
        # AllegroRelocateManagerEnv.__init__ once the trajectory is loaded).
        self.scene.robot = _build_robot_cfg()

        # Propagate task knobs into the action term.
        self.actions.arm_hand.cart_lin_vel_limit = self.task.cart_lin_vel_limit
        self.actions.arm_hand.cart_ang_vel_limit = self.task.cart_ang_vel_limit
        self.actions.arm_hand.ik_damping = self.task.ik_damping

        # Propagate task curriculum stage into the reset event.
        self.events.reset_traj.params["stage"] = self.task.stage

        # Propagate reward shaping into individual terms.
        self.rewards.pregrasp.params = {"err_scale": self.reward.pregrasp_err_scale}
        self.rewards.object_track.params = {
            "err_scale": self.reward.object_track_err_scale,
            "rot_weight": self.reward.object_rot_err_weight,
        }
        self.rewards.fingertip_track.params = {
            "err_scale": self.reward.fingertip_track_err_scale
        }
        self.rewards.lift_bonus.params = {"thresh": self.reward.lift_bonus_thresh}

        # Propagate object_too_far threshold.
        self.terminations.object_too_far.params = {"threshold": self.reward.obj_com_term}
