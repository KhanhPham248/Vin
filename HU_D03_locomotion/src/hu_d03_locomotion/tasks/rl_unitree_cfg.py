"""Cấu hình PPO runner theo Unitree cho HU_D03."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from hu_d03_locomotion.tasks.rl_cfg import (
    hu_d03_flat_ppo_runner_cfg as get_base_flat_runner_cfg,
    hu_d03_rough_ppo_runner_cfg as get_base_rough_runner_cfg,
)


def hu_d03_flat_unitree_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Cấu hình PPO cho môi trường mặt phẳng."""
    cfg = get_base_flat_runner_cfg()
    cfg.experiment_name = "hu_d03_flat_unitree"
    cfg.wandb_project = "hu_d03_locomotion_unitree"
    return cfg


def hu_d03_rough_unitree_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Cấu hình PPO cho môi trường gồ ghề ."""
    cfg = get_base_rough_runner_cfg()
    cfg.experiment_name = "hu_d03_rough_unitree"
    cfg.wandb_project = "hu_d03_locomotion_unitree"
    return cfg
