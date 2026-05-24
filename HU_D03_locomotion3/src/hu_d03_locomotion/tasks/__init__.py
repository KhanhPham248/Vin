"""Tasks subpackage — registers all HU_D03 tasks into mjlab's registry."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from hu_d03_locomotion.tasks.velocity_env_cfg import (
    hu_d03_flat_env_cfg,
    hu_d03_rough_env_cfg,
)
from hu_d03_locomotion.tasks.rl_cfg import (
    hu_d03_flat_ppo_runner_cfg,
    hu_d03_rough_ppo_runner_cfg,
)

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
