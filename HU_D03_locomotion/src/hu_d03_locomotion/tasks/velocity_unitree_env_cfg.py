from __future__ import annotations

import copy
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from hu_d03_locomotion.tasks import mdp_unitree
from hu_d03_locomotion.tasks.velocity_env_cfg import (
    hu_d03_flat_env_cfg as get_base_flat_cfg,
    hu_d03_rough_env_cfg as get_base_rough_cfg,
    ACTUATED_JOINT_NAMES,
)


def _apply_unitree_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    cfg = copy.deepcopy(cfg)

    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
        func=mdp_unitree.phase,
        params={"period": 0.65, "command_name": "twist"},
    )
    cfg.observations["critic"].terms["phase"] = ObservationTermCfg(
        func=mdp_unitree.phase,
        params={"period": 0.65, "command_name": "twist"},
    )

    cfg.rewards.pop("air_time", None)
    cfg.rewards.pop("foot_swing_height", None)
    cfg.rewards.pop("foot_slip", None)

    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=mdp_unitree.feet_gait,
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

    cfg.rewards["stand_still"] = RewardTermCfg(
        func=mdp_unitree.stand_still,
        weight=-1.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES),
        }
    )

    return cfg


def hu_d03_flat_unitree_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Cấu hình môi trường phẳng với chu kỳ dáng đi Unitree."""
    base_cfg = get_base_flat_cfg(play=play)
    return _apply_unitree_overrides(base_cfg)


def hu_d03_rough_unitree_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Cấu hình môi trường gồ ghề với chu kỳ dáng đi Unitree."""
    base_cfg = get_base_rough_cfg(play=play)
    return _apply_unitree_overrides(base_cfg)
