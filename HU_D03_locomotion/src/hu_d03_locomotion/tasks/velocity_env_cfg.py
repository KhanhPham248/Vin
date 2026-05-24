"""HU_D03 velocity environment configurations.

Extends mjlab's base velocity task (from G1) with HU_D03-specific:
- Achilles ankle linkage (policy controls A/B joints, not ankle_pitch/roll)
- base_link as root body (G1 uses pelvis)
- Single box foot geom + added sites for height/slip sensors
- Waist 4-bar linkage (waist_A/B_joint)
"""

import math
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    TerrainHeightSensorCfg,
    RingPatternCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from hu_d03_locomotion.robots import HU_D03_ACTION_SCALE, get_hu_d03_robot_cfg


# Shared constants
# ---------------------------------------------------------------------------

# Sites added to the XML for foot clearance / slip rewards
FOOT_SITE_NAMES = ("left_foot", "right_foot")

# Foot geom names (single box per foot in HU_D03)
FOOT_GEOM_NAMES = ("left_foot", "right_foot")

# Controllable (actuated) joints for pose penalty (excludes passive ankle and rod joints)
ACTUATED_JOINT_NAMES = (
    r".*_hip_pitch_joint",
    r".*_hip_roll_joint",
    r".*_hip_yaw_joint",
    r".*_knee_joint",
    r".*_A_achilles_joint",
    r".*_B_achilles_joint",
    r"waist_yaw_joint",
    r"waist_A_joint",
    r"waist_B_joint",
    r".*_shoulder_pitch_joint",
    r".*_shoulder_roll_joint",
    r".*_shoulder_yaw_joint",
    r".*_elbow_joint",
    r".*_wrist_yaw_joint",
    r".*_wrist_pitch_joint",
    r".*_hand_yaw_joint",
    r"head_yaw_joint",
    r"head_pitch_joint",
)


def _get_optimal_solver() -> str:
    """Determine the optimal solver based on environment variables and GPU capability.
    Older GPUs (e.g., Pascal sm_61) require the CG solver to avoid tiled matmul LTO failures.
    """
    import os
    if "MJLAB_SOLVER" in os.environ:
        return os.environ["MJLAB_SOLVER"]
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 7:  # Volta (sm_70) is the minimum required for Warp tile_matmul LTO
                return "cg"
    except Exception:
        pass
    return "newton"


# ---------------------------------------------------------------------------
# Flat terrain config (start here — fewer variables, fastest to debug)
# ---------------------------------------------------------------------------

def hu_d03_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """HU_D03 flat-terrain velocity tracking."""

    cfg = make_velocity_env_cfg()

    # ── Simulation tweaks ─────────────────────────────────────────────────
    cfg.sim.mujoco.ccd_iterations = 200
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None          # auto
    cfg.sim.mujoco.solver = _get_optimal_solver()

    # ── Robot ─────────────────────────────────────────────────────────────
    cfg.scene.entities = {"robot": get_hu_d03_robot_cfg()}

    # ── Flat terrain (no raycast needed) ──────────────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove terrain_scan raycast sensor (not needed on flat)
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    # Remove height_scan observation terms that rely on raycast
    for grp in ("actor", "critic"):
        cfg.observations[grp].terms.pop("height_scan", None)

    # ── Foot contact sensor ───────────────────────────────────────────────
    # HU_D03 ankle chain ends at ankle_roll_link — match that subtree
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Foot height scan sensor (uses sites we added to the XML)
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

    # Self-collision detector (pelvis subtree vs itself)
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    # ── Action scale (robot-specific) ─────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = HU_D03_ACTION_SCALE

    # ── Viewer ────────────────────────────────────────────────────────────
    cfg.viewer.body_name = "base_link"   # G1 uses "torso_link"

    # ── Velocity command display offset ───────────────────────────────────
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.1   # HU_D03 slightly shorter than G1
    twist_cmd.ranges.lin_vel_x = (0.0, 1.2)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    # ── Domain randomization ─────────────────────────────────────────────
    # Foot friction DR — HU_D03 has a single box geom per foot
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
    # CoM offset randomization on root body
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    # Disable push_robot to let the robot learn to stand first
    cfg.events.pop("push_robot", None)

    # ── Reward: upright body reference ───────────────────────────────────
    # Use waist_pitch_link (closest to torso in HU_D03 chain)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("waist_pitch_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("waist_pitch_link",)

    # ── Reward: foot clearance / slip — use XML sites ─────────────────────
    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    # ── Reward: pose std — tuned for HU_D03 joint topology ───────────────
    # Exclude passive ankle and rod joints to avoid conflicting/corrupting gradient penalties
    cfg.rewards["pose"].params["asset_cfg"].joint_names = ACTUATED_JOINT_NAMES
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body (actuated only)
        r".*hip_pitch.*": 0.30,
        r".*hip_roll.*":  0.15,
        r".*hip_yaw.*":   0.15,
        r".*knee.*":      0.35,
        # Achilles — looser to allow natural ankle motion
        r".*achilles.*":  0.20,
        # Waist linkage
        r"waist_yaw.*":   0.20,
        r"waist_roll.*":  0.15,
        r"waist_pitch.*": 0.20,
        r"waist_[AB].*":  0.15,
        # Arms — strict penalty to prevent dangling
        r".*shoulder.*":  0.05,
        r".*elbow.*":     0.05,
        r".*wrist.*":     0.10,
        r".*hand.*":      0.10,
        r".*head.*":      0.10,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*hip_pitch.*": 0.50,
        r".*hip_roll.*":  0.20,
        r".*hip_yaw.*":   0.20,
        r".*knee.*":      0.60,
        r".*achilles.*":  0.35,
        r"waist_yaw.*":   0.30,
        r"waist_roll.*":  0.20,
        r"waist_pitch.*": 0.30,
        r"waist_[AB].*":  0.20,
        r".*shoulder.*":  0.10,
        r".*elbow.*":     0.10,
        r".*wrist.*":     0.20,
        r".*hand.*":      0.20,
        r".*head.*":      0.15,
    }

    # Tighten velocity tracking standard deviation to prevent reward leaking and local optima convergence
    cfg.rewards["track_linear_velocity"].params["std"] = 0.15
    cfg.rewards["track_angular_velocity"].params["std"] = 0.25
    cfg.rewards["track_linear_velocity"].weight = 3.0

    cfg.rewards["foot_clearance"].weight = 0.0  # Tắt ở Flat, chỉ bật ở Rough
    cfg.rewards["action_rate_l2"].weight = -0.01 # Giảm phạt l2 để dễ cử động hơn
    cfg.rewards["soft_landing"].weight = -0.05
    cfg.rewards["foot_slip"].weight = -0.05
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02 # Trả về mức chuẩn của G1
    cfg.rewards["air_time"].weight = 3.0   # Enabled to encourage stepping (HU-D03 needs this)
    cfg.rewards["air_time"].params["command_threshold"] = 0.1  # Fix: Kích hoạt thưởng bay chân ngay cả khi đi chậm (vận tốc > 0.1)

    # ── Self-collision penalty ─────────────────────────────────────────────
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    )

    # ── Fell Over Penalty ──────────────────────────────────────────────────
    cfg.rewards["fell_over_penalty"] = RewardTermCfg(
        func=mdp.bad_orientation,
        weight=0.0,  # Tắt hẳn phạt ngã ở giai đoạn đầu để robot dám bước đi
        params={"limit_angle": math.radians(85.0)},
    )

    # ── Curriculum (walking up to 1.2 m/s) ───────────────────────────
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (0.0, 0.5), "ang_vel_z": (0.0, 0.0)},
                {"step": 2000 * 24, "lin_vel_x": (0.0, 1.0), "ang_vel_z": (0.0, 0.0)},
                {"step": 4000 * 24, "lin_vel_x": (0.0, 1.2), "ang_vel_z": (0.0, 0.0)},
            ],
        },
    )
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(85.0)

    # ── Play mode overrides ───────────────────────────────────────────────
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg


# ---------------------------------------------------------------------------
# Rough terrain config (use after flat training succeeds)
# ---------------------------------------------------------------------------

def hu_d03_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """HU_D03 rough-terrain velocity tracking (adds height scan + curriculum)."""

    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 70
    cfg.sim.mujoco.solver = _get_optimal_solver()

    cfg.scene.entities = {"robot": get_hu_d03_robot_cfg()}

    # Raycast sensor frame → base_link (G1 uses pelvis)
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "base_link"

        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = HU_D03_ACTION_SCALE

    cfg.viewer.body_name = "base_link"
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("waist_pitch_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("waist_pitch_link",)
    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    cfg.rewards["pose"].params["asset_cfg"].joint_names = ACTUATED_JOINT_NAMES
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_pitch.*": 0.30, r".*hip_roll.*": 0.15, r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,      r".*achilles.*": 0.20,
        r"waist_yaw.*": 0.20,   r"waist_roll.*": 0.15, r"waist_pitch.*": 0.20,
        r"waist_[AB].*": 0.15,
        r".*shoulder.*": 0.05,  r".*elbow.*": 0.05,
        r".*wrist.*": 0.10,     r".*hand.*": 0.10, r".*head.*": 0.10,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*hip_pitch.*": 0.50, r".*hip_roll.*": 0.20, r".*hip_yaw.*": 0.20,
        r".*knee.*": 0.60,      r".*achilles.*": 0.35,
        r"waist_yaw.*": 0.30,   r"waist_roll.*": 0.20, r"waist_pitch.*": 0.30,
        r"waist_[AB].*": 0.20,
        r".*shoulder.*": 0.10,  r".*elbow.*": 0.10,
        r".*wrist.*": 0.20,     r".*hand.*": 0.20, r".*head.*": 0.15,
    }
    cfg.rewards["track_linear_velocity"].params["std"] = 0.15
    cfg.rewards["track_angular_velocity"].params["std"] = 0.25
    cfg.rewards["foot_clearance"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["soft_landing"].weight = -0.05
    cfg.rewards["foot_slip"].weight = -0.05
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.1
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    )

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 1.2)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    cfg.curriculum.pop("command_vel", None)
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(85.0)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5
            cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg
