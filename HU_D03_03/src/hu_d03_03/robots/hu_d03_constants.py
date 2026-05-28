""""""

import math
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


HU_D03_XML: Path = Path(__file__).parent.parent.parent.parent / "assets" / "robots" / "hu_d03" / "xmls" / "hu_d03.xml"
assert HU_D03_XML.exists(), f"Không tìm thấy XML của HU_D03: {HU_D03_XML}"


def get_spec() -> mujoco.MjSpec:
    """"""

    spec = mujoco.MjSpec.from_file(str(HU_D03_XML))

    terrain_bit = 1
    robot_self_bit = 4
    feet_bits = terrain_bit | robot_self_bit

    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            continue

        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.contype = 0
            geom.conaffinity = 0
            continue

        if geom.name in ("left_foot", "right_foot"):
            geom.condim = 3
            geom.contype = feet_bits
            geom.conaffinity = feet_bits
            geom.friction[0] = 0.8
        else:
            geom.condim = 3
            geom.contype = robot_self_bit
            geom.conaffinity = robot_self_bit

    return spec


LEG_NATURAL_FREQ = 8.0 * 2.0 * math.pi
ARM_NATURAL_FREQ = 4.0 * 2.0 * math.pi

DAMPING_RATIO = 1.5

ARM_HIP_KNEE  = 0.15257125
ARM_ACHILLES  = 0.094889232
ARM_SHOULDER  = 0.045760625
ARM_WRIST     = 0.010625

def _gains(armature: float, freq: float) -> tuple[float, float]:
    """"""
    kp = armature * freq ** 2
    kd = 2.0 * DAMPING_RATIO * armature * freq
    return kp, kd

_kp_hk, _kd_hk = _gains(ARM_HIP_KNEE, LEG_NATURAL_FREQ)
_kp_ac, _kd_ac = _gains(ARM_ACHILLES, LEG_NATURAL_FREQ)
_kp_sh, _kd_sh = _gains(ARM_SHOULDER, ARM_NATURAL_FREQ)
_kp_wr, _kd_wr = _gains(ARM_WRIST, ARM_NATURAL_FREQ)


HU_D03_ACTUATOR_HIP_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hip_pitch_joint",
        ".*_hip_roll_joint",
        ".*_hip_yaw_joint",
        ".*_knee_joint",
    ),
    stiffness=_kp_hk,
    damping=_kd_hk,
    effort_limit=120.0,
    armature=ARM_HIP_KNEE,
)

HU_D03_ACTUATOR_ACHILLES = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_A_achilles_joint",
        ".*_B_achilles_joint",
    ),
    stiffness=_kp_ac,
    damping=_kd_ac,
    effort_limit=45.0,
    armature=ARM_ACHILLES,
)

HU_D03_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "waist_yaw_joint",
        "waist_A_joint",
        "waist_B_joint",
    ),
    stiffness=_kp_ac,
    damping=_kd_ac,
    effort_limit=45.0,
    armature=ARM_ACHILLES,
)

HU_D03_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
    ),
    stiffness=_kp_sh,
    damping=_kd_sh,
    effort_limit=30.0,
    armature=ARM_SHOULDER,
)

HU_D03_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_wrist_yaw_joint",
        ".*_wrist_pitch_joint",
        ".*_hand_yaw_joint",
        "head_yaw_joint",
        "head_pitch_joint",
    ),
    stiffness=_kp_wr,
    damping=_kd_wr,
    effort_limit=18.0,
    armature=ARM_WRIST,
)

HU_D03_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        HU_D03_ACTUATOR_HIP_KNEE,
        HU_D03_ACTUATOR_ACHILLES,
        HU_D03_ACTUATOR_WAIST,
        HU_D03_ACTUATOR_SHOULDER,
        HU_D03_ACTUATOR_WRIST,
    ),
    soft_joint_pos_limit_factor=0.9,
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.87),
    joint_pos={
        ".*_hip_pitch_joint":  -0.10,
        ".*_hip_roll_joint":    0.00,
        ".*_hip_yaw_joint":     0.00,
        ".*_knee_joint":        0.30,
        ".*_A_achilles_joint":  0.00,
        ".*_B_achilles_joint":  0.00,
        "waist_yaw_joint":      0.00,
        "waist_A_joint":        0.00,
        "waist_B_joint":        0.00,
        ".*_shoulder_pitch_joint": 0.10,
        "left_shoulder_roll_joint": 0.20,
        "right_shoulder_roll_joint": -0.20,
        ".*_shoulder_yaw_joint":   0.00,
        ".*_elbow_joint":          -0.80,
        ".*_wrist_yaw_joint":      0.00,
        ".*_wrist_pitch_joint":    0.00,
        ".*_hand_yaw_joint":       0.00,
        "head_yaw_joint":          0.00,
        "head_pitch_joint":        0.00,
    },
    joint_vel={".*": 0.0},
)


HU_D03_ACTION_SCALE: dict[str, float] = {}
for _act in HU_D03_ARTICULATION.actuators:
    assert isinstance(_act, BuiltinPositionActuatorCfg)
    e = _act.effort_limit
    s = _act.stiffness
    assert e is not None
    
    is_leg = any(name in _act.target_names_expr[0] for name in ["hip", "knee", "achilles"])
    factor = 1.0 if is_leg else 0.25
    
    for _n in _act.target_names_expr:
        HU_D03_ACTION_SCALE[_n] = factor * e / s


def get_hu_d03_robot_cfg() -> EntityCfg:
    """"""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(),
        spec_fn=get_spec,
        articulation=HU_D03_ARTICULATION,
    )
