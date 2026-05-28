"""Tasks subpackage — registers all HU_D03 tasks into mjlab's registry."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from hu_d03_03.tasks.velocity.velocity_env_cfg import (
    hu_d03_flat_env_cfg,
    hu_d03_rough_env_cfg,
)
from hu_d03_03.tasks.velocity.rl_cfg import (
    hu_d03_flat_ppo_runner_cfg,
    hu_d03_rough_ppo_runner_cfg,
)
from hu_d03_03.tasks.mimic.mimic_env_cfg import hu_d03_mimic_flat_env_cfg
from hu_d03_03.tasks.mimic.rl_cfg import hu_d03_mimic_ppo_runner_cfg
from hu_d03_03.tasks.mimic.rl.runner import MotionTrackingOnPolicyRunner

# ── Standard Tasks ──────────────────────────────────────────────────────────

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-HuD03",
    env_cfg=hu_d03_flat_env_cfg(),
    play_env_cfg=hu_d03_flat_env_cfg(play=True),
    rl_cfg=hu_d03_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-HuD03",
    env_cfg=hu_d03_rough_env_cfg(),
    play_env_cfg=hu_d03_rough_env_cfg(play=True),
    rl_cfg=hu_d03_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# ── Mimic Tasks ─────────────────────────────────────────────────────────────

register_mjlab_task(
    task_id="Mjlab-Mimic-Flat-HuD03",
    env_cfg=hu_d03_mimic_flat_env_cfg(),
    play_env_cfg=hu_d03_mimic_flat_env_cfg(play=True),
    rl_cfg=hu_d03_mimic_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
