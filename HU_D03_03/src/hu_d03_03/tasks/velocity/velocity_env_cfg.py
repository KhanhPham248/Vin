import math
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
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

from hu_d03_03.tasks.velocity import mdp_gait

from hu_d03_03.robots import HU_D03_ACTION_SCALE, get_hu_d03_robot_cfg

FOOT_SITE_NAMES = ("left_foot", "right_foot")
FOOT_GEOM_NAMES = ("left_foot", "right_foot")

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
    """"""
    import os
    if "MJLAB_SOLVER" in os.environ:
        return os.environ["MJLAB_SOLVER"]
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 7:
                return "cg"
    except Exception:
        pass
    return "newton"


def hu_d03_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """"""

    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = 64
    cfg.sim.mujoco.solver = _get_optimal_solver()

    # Robot
    cfg.scene.entities = {"robot": get_hu_d03_robot_cfg()}

    actuated_asset_cfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)
    for grp in ("actor", "critic"):
        cfg.observations[grp].terms["joint_pos"].params["asset_cfg"] = actuated_asset_cfg
        cfg.observations[grp].terms["joint_vel"].params["asset_cfg"] = actuated_asset_cfg

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    for grp in ("actor", "critic"):
        cfg.observations[grp].terms.pop("height_scan", None)
    cfg.observations["actor"].terms.pop("base_lin_vel", None)

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

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

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

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = HU_D03_ACTION_SCALE

    cfg.viewer.body_name = "base_link"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.1
    twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    
    twist_cmd.heading_command = False
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.rel_forward_envs = 1.0
    twist_cmd.ranges.heading = None

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    cfg.events.pop("push_robot", None)

    if "upright" in cfg.rewards:
        cfg.rewards.pop("upright")
    cfg.rewards["flat_orientation_l2"] = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("waist_pitch_link",))}
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("waist_pitch_link",)

    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    cfg.rewards["pose"].params["asset_cfg"].joint_names = ACTUATED_JOINT_NAMES
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["walking_threshold"] = 0.2
    cfg.rewards["pose"].params["running_threshold"] = 1.5
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_pitch.*": 0.50,  
        r".*hip_roll.*":  0.15,
        r".*hip_yaw.*":   0.15,
        r".*knee.*":      0.50,
        r".*achilles.*":  0.20,
        r"waist_yaw.*":   0.10,
        r"waist_[AB].*":  0.15,
        r".*shoulder.*":  0.10,
        r".*elbow.*":     0.10,
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
        r"waist_[AB].*":  0.20,
        r".*shoulder.*":  0.05,
        r".*elbow.*":     0.05,
        r".*wrist.*":     0.10,
        r".*hand.*":      0.10,
        r".*head.*":      0.10,
    }

    
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
    cfg.rewards["track_linear_velocity"].weight = 1.0
    cfg.rewards["track_angular_velocity"].weight = 1.0

    cfg.rewards["pose"].weight = 1.0
    cfg.rewards["foot_clearance"].weight = -1.0  
    cfg.rewards["foot_clearance"].params["target_height"] = 0.10
    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=mdp_gait.feet_gait,
        weight=0.5,
        params={
            "period": 0.65,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        }
    )

    cfg.rewards["is_terminated"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-150.0
    )
    cfg.rewards["feet_stuck"] = RewardTermCfg(
        func=mdp_gait.feet_stuck_penalty,
        weight=-2.0,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact", "command_threshold": 0.1}
    )

    cfg.rewards["joint_pos_limits"] = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    cfg.rewards["action_rate_l2"].weight = -0.05 
    cfg.rewards["soft_landing"].weight = -0.001
    
    cfg.rewards["body_ang_vel"].weight = -0.1
    cfg.rewards["angular_momentum"].weight = -0.01

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    )

    cfg.rewards["arm_velocity_penalty"] = RewardTermCfg(
        func=mdp_gait.arm_velocity_penalty,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"))}
    )

    cfg.rewards["stand_still"] = RewardTermCfg(
        func=mdp_gait.stand_still,
        weight=-1.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)
        }
    )

    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-0.5, 1.0), "ang_vel_z": (0.0, 0.0)},
                {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "ang_vel_z": (0.0, 0.0)},
            ],
        },
    )
    cfg.terminations.pop("out_of_terrain_bounds", None)
    limit = math.radians(70.0)
    cfg.terminations["fell_over"].params["limit_angle"] = limit

    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
        func=mdp_gait.phase,
        params={"period": 0.65, "command_name": "twist"},
    )
    cfg.observations["critic"].terms["phase"] = ObservationTermCfg(
        func=mdp_gait.phase,
        params={"period": 0.65, "command_name": "twist"},
    )

    cfg.rewards.pop("air_time", None)
    cfg.rewards.pop("foot_swing_height", None)
    cfg.rewards.pop("foot_slip", None)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        twist_cmd.ranges.lin_vel_x = (0.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg


def hu_d03_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """"""

    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.nconmax = 128
    cfg.sim.mujoco.solver = _get_optimal_solver()

    cfg.scene.entities = {"robot": get_hu_d03_robot_cfg()}

    actuated_asset_cfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)
    for grp in ("actor", "critic"):
        cfg.observations[grp].terms["joint_pos"].params["asset_cfg"] = actuated_asset_cfg
        cfg.observations[grp].terms["joint_vel"].params["asset_cfg"] = actuated_asset_cfg
    cfg.observations["actor"].terms.pop("base_lin_vel", None)

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
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("waist_pitch_link",)
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    cfg.rewards["pose"].params["asset_cfg"].joint_names = ACTUATED_JOINT_NAMES
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["walking_threshold"] = 0.1
    cfg.rewards["pose"].params["running_threshold"] = 1.5
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_pitch.*": 0.50, r".*hip_roll.*": 0.15, r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.50,      r".*achilles.*": 0.20,
        r"waist_yaw.*": 0.10,   r"waist_[AB].*": 0.15,
        r".*shoulder.*": 0.10,  r".*elbow.*": 0.10,
        r".*wrist.*": 0.10,     r".*hand.*": 0.10, r".*head.*": 0.10,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*hip_pitch.*": 0.50, r".*hip_roll.*": 0.20, r".*hip_yaw.*": 0.20,
        r".*knee.*": 0.60,      r".*achilles.*": 0.35,
        r"waist_yaw.*": 0.30,   r"waist_[AB].*": 0.20,
        r".*shoulder.*": 0.05,  r".*elbow.*": 0.05,
        r".*wrist.*": 0.10,     r".*hand.*": 0.10, r".*head.*": 0.10,
    }
    
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
    cfg.rewards["track_linear_velocity"].weight = 1.0
    cfg.rewards["track_angular_velocity"].weight = 1.0

    cfg.rewards["pose"].weight = 1.0
    cfg.rewards["foot_clearance"].weight = -1.0
    cfg.rewards["foot_clearance"].params["target_height"] = 0.10
    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=mdp_gait.feet_gait,
        weight=0.5,
        params={
            "period": 0.65,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        }
    )

    cfg.rewards["is_terminated"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-150.0
    )
    cfg.rewards["feet_stuck"] = RewardTermCfg(
        func=mdp_gait.feet_stuck_penalty,
        weight=-2.0,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact", "command_threshold": 0.1}
    )
    cfg.rewards["joint_pos_limits"] = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["soft_landing"].weight = -0.001

    cfg.rewards.pop("upright", None)
    cfg.rewards["flat_orientation_l2"] = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("waist_pitch_link",))}
    )
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    )

    cfg.rewards["arm_velocity_penalty"] = RewardTermCfg(
        func=mdp_gait.arm_velocity_penalty,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"))}
    )

    cfg.rewards["stand_still"] = RewardTermCfg(
        func=mdp_gait.stand_still,
        weight=-1.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES)
        }
    )

    limit = math.radians(70.0)

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.2, 1.2)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    cfg.curriculum.pop("command_vel", None)
    cfg.terminations["fell_over"].params["limit_angle"] = limit

    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
        func=mdp_gait.phase,
        params={"period": 0.65, "command_name": "twist"},
    )
    cfg.observations["critic"].terms["phase"] = ObservationTermCfg(
        func=mdp_gait.phase,
        params={"period": 0.65, "command_name": "twist"},
    )

    cfg.rewards.pop("air_time", None)
    cfg.rewards.pop("foot_swing_height", None)
    cfg.rewards.pop("foot_slip", None)

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
