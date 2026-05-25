import sys
import os

# add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append('/home/khanh248/Documents/Vin/mjlab/mjlab-main/src')

from hu_d03_locomotion.tasks.velocity_unitree_env_cfg import hu_d03_flat_unitree_env_cfg
from mjlab.envs import ManagerBasedRlEnv

env_cfg = hu_d03_flat_unitree_env_cfg()
env_cfg.scene.num_envs = 1
try:
    env = ManagerBasedRlEnv(cfg=env_cfg)
    obs = env.reset()
    actor_obs = obs[0]["actor"]
    print("Actor obs shape:", actor_obs.shape)
    
    # Print individual observation term sizes
    for term_name, term_val in env.observation_manager.group_obs["actor"].items():
        print(f"Term '{term_name}': {term_val.shape}")
        
    print(f"Num actuated joints: {len(env.scene['robot'].data.actuator_joint_names)}")
    print(f"Actuator names: {env.scene['robot'].data.actuator_joint_names}")
    
except Exception as e:
    import traceback
    traceback.print_exc()

