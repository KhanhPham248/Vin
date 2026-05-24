"""HU_D03 robot constants — PD gains, keyframe, collision config, EntityCfg."""

import math
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HU_D03_XML: Path = Path(__file__).parent.parent.parent.parent / "assets" / "robots" / "hu_d03" / "xmls" / "hu_d03.xml"
assert HU_D03_XML.exists(), f"HU_D03 XML not found: {HU_D03_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(HU_D03_XML))


# ---------------------------------------------------------------------------
# PD Gain Design
#
# Formula (same as G1):
#   stiffness = armature * NATURAL_FREQ^2
#   damping   = 2 * DAMPING_RATIO * armature * NATURAL_FREQ
#
# Armatures from XML:
#   hip / knee joints:   0.15257125
#   achilles joints:     0.094889232
#   waist A/B joints:    0.094889232
#   shoulder joints:     0.045760625
#   wrist / head joints: 0.010625
# ---------------------------------------------------------------------------

NATURAL_FREQ = 10.0 * 2.0 * math.pi   # 10 Hz
DAMPING_RATIO = 2.0

ARM_HIP_KNEE  = 0.15257125
ARM_ACHILLES  = 0.094889232
ARM_SHOULDER  = 0.045760625
ARM_WRIST     = 0.010625

def _gains(armature: float) -> tuple[float, float]:
    kp = armature * NATURAL_FREQ ** 2
    kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
    return kp, kd

_kp_hk, _kd_hk = _gains(ARM_HIP_KNEE)    # hip + knee
_kp_ac, _kd_ac = _gains(ARM_ACHILLES)     # achilles + waist A/B
_kp_sh, _kd_sh = _gains(ARM_SHOULDER)     # shoulder/elbow
_kp_wr, _kd_wr = _gains(ARM_WRIST)        # wrist + head

# ---------------------------------------------------------------------------
# Effort limits from XML actuator ctrlrange
# ---------------------------------------------------------------------------

# Hip / knee: ±120 Nm  |  Achilles / waist: ±45 Nm
# Shoulder / elbow: ±30 Nm  |  Wrist / hand: ±18 Nm  |  Head: ±18 Nm

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

# Achilles joints drive the ankle via 4-bar linkage (ankle joints are passive)
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

# ---------------------------------------------------------------------------
# Keyframe — Standing Pose
#
# Height estimate:
#   hip → knee: 0.3045 m
#   knee → ankle_pitch: 0.3855 m
#   ankle_pitch → foot bottom: ~0.062 m
#   Total hip height ≈ 0.752 m
#   base_link above hip ≈ ~0.12 m (from geometry)
#   → base_link height ≈ 0.87 m
#
# Achilles joints (A/B) at 0.0 maps to ankle flat.
# NOTE: Calibrate these values with view_mujoco.py if robot doesn't stand correctly.
# ---------------------------------------------------------------------------

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.87),
    joint_pos={
        # Legs — slight knee bend for stability
        ".*_hip_pitch_joint":  -0.10,
        ".*_hip_roll_joint":    0.00,
        ".*_hip_yaw_joint":     0.00,
        ".*_knee_joint":        0.30,
        # Achilles: 0.0 = neutral ankle pitch/roll
        ".*_A_achilles_joint":  0.00,
        ".*_B_achilles_joint":  0.00,
        # Waist — neutral
        "waist_yaw_joint":      0.00,
        "waist_A_joint":        0.00,
        "waist_B_joint":        0.00,
        # Arms — G1 style (elbows bent forward, shoulders slightly out)
        ".*_shoulder_pitch_joint": 0.10,
        "left_shoulder_roll_joint": 0.20,
        "right_shoulder_roll_joint": -0.20,
        ".*_shoulder_yaw_joint":   0.00,
        ".*_elbow_joint":          -0.80,  # Âm là gập về trước đối với HU-D03
        ".*_wrist_yaw_joint":      0.00,
        ".*_wrist_pitch_joint":    0.00,
        ".*_hand_yaw_joint":       0.00,
        # Head
        "head_yaw_joint":          0.00,
        "head_pitch_joint":        0.00,
    },
    joint_vel={".*": 0.0},
)

# ---------------------------------------------------------------------------
# Collision config — only feet contact terrain/objects
# ---------------------------------------------------------------------------

HU_D03_COLLISION = CollisionCfg(
    geom_names_expr=("left_foot", "right_foot"),
    condim=3,
    friction=(0.6,),
)

# ---------------------------------------------------------------------------
# Action scale (Using fixed rad scales to prevent high stiffness paralysis)
# ---------------------------------------------------------------------------

HU_D03_ACTION_SCALE: dict[str, float] = {}
for _act in HU_D03_ARTICULATION.actuators:
    assert isinstance(_act, BuiltinPositionActuatorCfg)
    for _n in _act.target_names_expr:
        # Leg joints (hips, knees, ankles) and waist need realistic range (0.25 rad ≈ 14.3 deg)
        if any(keyword in _n for keyword in ("hip", "knee", "achilles", "waist")):
            HU_D03_ACTION_SCALE[_n] = 0.25
        else:
            HU_D03_ACTION_SCALE[_n] = 0.30


# ---------------------------------------------------------------------------
# Entity config factory
# ---------------------------------------------------------------------------

def get_hu_d03_robot_cfg() -> EntityCfg:
    """Return a fresh HU_D03 EntityCfg (call each time to avoid mutation)."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(HU_D03_COLLISION,),
        spec_fn=get_spec,
        articulation=HU_D03_ARTICULATION,
    )
