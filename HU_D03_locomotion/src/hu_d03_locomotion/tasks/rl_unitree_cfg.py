"""Unitree-Style PPO runner configurations for HU_D03.

Defines isolated experiment folders for side-by-side PPO comparison in TensorBoard.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from hu_d03_locomotion.tasks.rl_cfg import (
    hu_d03_flat_ppo_runner_cfg as get_base_flat_runner_cfg,
    hu_d03_rough_ppo_runner_cfg as get_base_rough_runner_cfg,
)


def hu_d03_flat_unitree_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO runner configuration for Flat Unitree-style environment."""
    cfg = get_base_flat_runner_cfg()
    cfg.experiment_name = "hu_d03_flat_unitree"
    cfg.wandb_project = "hu_d03_locomotion_unitree"
    return cfg


def hu_d03_rough_unitree_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO runner configuration for Rough Unitree-style environment."""
    cfg = get_base_rough_runner_cfg()
    cfg.experiment_name = "hu_d03_rough_unitree"
    cfg.wandb_project = "hu_d03_locomotion_unitree"
    return cfg
