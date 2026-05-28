"""Cấu hình PPO runner cho tác vụ bắt chước (Mimic / Tracking) của HU_D03."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from hu_d03_03.tasks.velocity.rl_cfg import hu_d03_flat_ppo_runner_cfg

def hu_d03_mimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Cấu hình PPO cho môi trường mimic/tracking."""
    cfg = hu_d03_flat_ppo_runner_cfg()
    cfg.algorithm.entropy_coef = 0.005
    cfg.experiment_name = "hu_d03_03_mimic"
    cfg.save_interval = 500
    cfg.max_iterations = 30_000
    return cfg
