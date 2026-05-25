"""Unitree-Style dynamic environment configurations for HU_D03.

This module layers Unitree's cyclic gait rewards, phase observations, and standing still
pose locks on top of the base HU_D03 task configurations for safe side-by-side comparison.
"""

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
    """Helper to inject Unitree-specific phase observations and rewards into a config."""
    # Clone the config to prevent modifying the base task in-place
    cfg = copy.deepcopy(cfg)

    # 1. Inject 'phase' Observation [sin(2pi * t/T), cos(2pi * t/T)]
    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
        func=mdp_unitree.phase,
        params={"period": 0.6, "command_name": "twist"},
    )
    cfg.observations["critic"].terms["phase"] = ObservationTermCfg(
        func=mdp_unitree.phase,
        params={"period": 0.6, "command_name": "twist"},
    )

    # 2. Disable generic foot air time reward in favor of strict phase gait matching
    cfg.rewards.pop("air_time", None)

    # 3. Add cyclic foot gait reward (Alternating Left/Right Stance/Swing)
    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=mdp_unitree.feet_gait,
        weight=0.5,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5],          # 180 degrees out of phase (Trot/Walk)
            "threshold": 0.56,             # Stance phase ratio
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        }
    )

    # 4. Add standing still joint lock (encourages rigid static standing posture)
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=mdp_unitree.stand_still,
        weight=-1.0,                       # Penalize posture errors at zero velocity
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES),
        }
    )

    # 5 & 6. contact_asymmetry and contact_duration_penalty are already injected
    # by the base flat/rough config (velocity_env_cfg.py). No need to re-declare here.

    return cfg


def hu_d03_flat_unitree_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """HU_D03 Flat-terrain configuration with Unitree-style cyclic gait."""
    base_cfg = get_base_flat_cfg(play=play)
    return _apply_unitree_overrides(base_cfg)


def hu_d03_rough_unitree_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """HU_D03 Rough-terrain configuration with Unitree-style cyclic gait."""
    base_cfg = get_base_rough_cfg(play=play)
    return _apply_unitree_overrides(base_cfg)
