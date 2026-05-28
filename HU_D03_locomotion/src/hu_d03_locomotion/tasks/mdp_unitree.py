""""""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def phase(
    env: ManagerBasedRlEnv, 
    period: float, 
    command_name: str
) -> torch.Tensor:
    """"""
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase_tensor = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
    
    phase_tensor[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase_tensor[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    
    command = env.command_manager.get_command(command_name)
    if command is not None:
        cmd_norm = torch.linalg.norm(command[:, :2], dim=1)
        smooth_scale = torch.clamp(cmd_norm / 0.1, 0.0, 1.0).unsqueeze(1)
        phase_tensor = phase_tensor * smooth_scale
        
    return phase_tensor


def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """"""
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    
    is_stance = (leg_phase < threshold)
    
    reward = (is_stance == is_contact).float().mean(dim=1)
    
    command = env.command_manager.get_command(command_name)
    if command is not None:
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        scale = (total_command > command_threshold).float()
        reward *= scale
        
    return reward


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """"""
    asset: Entity = env.scene[asset_cfg.name]
    
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    
    command = env.command_manager.get_command(command_name)
    if command is not None:
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        reward *= (total_command < command_threshold).float()
    return reward


def yaw_velocity_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """"""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    actual = asset.data.root_link_ang_vel_b
    return torch.square(command[:, 2] - actual[:, 2])


def arm_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """"""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def feet_stuck_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """"""
    sensor = env.scene.sensors[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    both_feet_down = torch.all(is_contact, dim=1)
    
    command = env.command_manager.get_command(command_name)
    linear_norm = torch.norm(command[:, :2], dim=1)
    
    return torch.logical_and(both_feet_down, linear_norm > command_threshold).float()

