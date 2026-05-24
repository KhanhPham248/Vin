from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import hu_d03_flat_env_cfg
from .rl_cfg import hu_d03_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HU-D03",
  env_cfg=hu_d03_flat_env_cfg(),
  play_env_cfg=hu_d03_flat_env_cfg(play=True),
  rl_cfg=hu_d03_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
