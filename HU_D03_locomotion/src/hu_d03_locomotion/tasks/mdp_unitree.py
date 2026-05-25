"""Custom MDP terms (rewards and observations) imported from Unitree RL for HU_D03."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ── Observation: Cyclic Phase Coordinate ────────────────────────────────────

def phase(
    env: ManagerBasedRlEnv, 
    period: float, 
    command_name: str
) -> torch.Tensor:
    """Cyclic phase observation term.
    
    Generates [sin(2pi * t/T), cos(2pi * t/T)] to provide a stable clock signal
    for gait synchronization. The signal is zeroed out if the commanded velocity
    is zero (robot stands still).
    """
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase_tensor = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
    
    # Calculate sin and cos of the phase angle
    phase_tensor[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase_tensor[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    
    # Zero out the phase if no motion command is active
    command = env.command_manager.get_command(command_name)
    if command is not None:
        stand_mask = torch.linalg.norm(command[:, :2], dim=1) < 0.1
        phase_tensor = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase_tensor), phase_tensor)
        
    return phase_tensor


# ── Reward: Foot Gait Synchronization ───────────────────────────────────────

def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Encourage an alternating stance and swing pattern synchronized with global phase."""
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    
    # Compute phase coordinate per environment
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    
    # Leg is in stance if phase is below the threshold
    is_stance = (leg_phase < threshold)
    
    # Reward matching contact status
    reward = (is_stance == is_contact).float().mean(dim=1)
    
    # Only active when robot is commanded to move
    command = env.command_manager.get_command(command_name)
    if command is not None:
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        scale = (total_command > command_threshold).float()
        reward *= scale
        
    return reward


# ── Reward: Quiet Stand Still Posture ───────────────────────────────────────

def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviations from default pose only when commanded velocity is near zero."""
    asset: Entity = env.scene[asset_cfg.name]
    
    # Compute L2 error between current joint positions and home keyframe positions
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    
    # Only active when robot is commanded to stand still (command <= threshold)
    command = env.command_manager.get_command(command_name)
    if command is not None:
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        scale = (total_command <= command_threshold).float()
        reward *= scale
        
    return reward


# ── Reward: Foot Contact Asymmetry Penalty ───────────────────────────────────

def contact_asymmetry(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize asymmetric foot contact duration between left and right feet."""
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time # Shape: [num_envs, 2] (left, right)
    
    # Absolute difference in contact time between Left and Right feet
    asym = torch.abs(contact_time[:, 0] - contact_time[:, 1])
    
    # Raw magnitude (positive)
    penalty = asym
    
    # Only active when commanded to move
    command = env.command_manager.get_command(command_name)
    if command is not None:
        total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        scale = (total_command > command_threshold).float()
        penalty *= scale
        
    return penalty


# ── Reward: Contact Duration Penalty (Peg-Leg Prevention) ─────────────────────

def contact_duration_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    max_duration: float = 0.6,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize feet that remain in contact with the ground for too long when moving."""
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time # Shape: [num_envs, 2]
    
    # Penalize any foot whose contact time exceeds max_duration
    excess_time = torch.clamp(contact_time - max_duration, min=0.0)
    penalty = torch.sum(excess_time, dim=1)
    
    # Only active when commanded to move
    command = env.command_manager.get_command(command_name)
    if command is not None:
        total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        scale = (total_command > command_threshold).float()
        penalty *= scale
        
    return penalty


# ── Reward: Touchdown-only Air Time Reward ───────────────────────────────────

def feet_air_time_touchdown(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str | None = None,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward feet air time ONLY at the moment of touchdown.
    
    This completely prevents "air-rowing" or keeping one foot in the air indefinitely
    since a foot must actively land to receive the reward.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    
    # Detect touchdown: foot has just landed in this step (contact started within env.step_dt)
    touchdown = sensor.compute_first_contact(env.step_dt)
    
    # Get the duration of the completed air phase
    air_time = sensor.data.last_air_time
    
    # Only reward if the completed air time is between threshold_min and threshold_max
    # This prevents extremely long, floaty steps or keeping a foot in the air indefinitely
    reward_per_foot = torch.clamp(air_time - threshold_min, min=0.0) * (air_time <= threshold_max).float() * touchdown.float()
    
    reward = torch.sum(reward_per_foot, dim=1)
    
    # Only active when commanded to move
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            scale = (total_command > command_threshold).float()
            reward *= scale
            
    return reward


# ── Termination: Undesired Contacts ──────────────────────────────────────────

def undesired_contacts(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate the episode if any undesired part of the robot touches the ground."""
    sensor: ContactSensor = env.scene[sensor_name]
    
    # net_force_norm shape: [num_envs, num_frames] -> check max force across all tracked frames
    max_contact_force = torch.max(sensor.data.net_force_norm, dim=1)[0]
    
    return max_contact_force > threshold
