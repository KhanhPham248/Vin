"""HU_D03 constants.

This is a *direct-drive training* configuration:
- Uses programmatically created MuJoCo built-in position actuators (PD).
- The MJCF asset itself intentionally has no <actuator> block.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

#
# MJCF and assets.
#

HU_D03_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "hu_d03" / "xmls" / "hu_d03_direct_drive.xml"
)
assert HU_D03_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(HU_D03_XML))


#
# Actuator config.
#

# A simple, conservative PD setup for initial training.
# (Can be tuned later once the task is stable.)
_HU_D03_STIFFNESS = 80.0
_HU_D03_DAMPING = 8.0
_HU_D03_EFFORT_LIMIT = 250.0

HU_D03_ACTUATOR_GROUP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_hand_yaw_joint",
  ),
  stiffness=_HU_D03_STIFFNESS,
  damping=_HU_D03_DAMPING,
  effort_limit=_HU_D03_EFFORT_LIMIT,
)

HU_D03_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(HU_D03_ACTUATOR_GROUP,),
  soft_joint_pos_limit_factor=0.9,
)

#
# Keyframe config.
#

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  # Estimated such that left/right foot sites are near z=0.
  pos=(0.0, 0.0, 0.88),
  joint_pos={
    ".*_hip_pitch_joint": -0.3,
    ".*_knee_joint": 0.6,
    ".*_ankle_pitch_joint": -0.3,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)


def get_hu_d03_robot_cfg() -> EntityCfg:
  """Get a fresh HU_D03 robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    spec_fn=get_spec,
    articulation=HU_D03_ARTICULATION,
  )


#
# Action scale.
#

# JointPositionAction uses: q_target = q_default + action * scale
# Keep this modest for early stability.
HU_D03_ACTION_SCALE: dict[str, float] = {".*": 0.30}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_hu_d03_robot_cfg())
  viewer.launch(robot.spec.compile())
