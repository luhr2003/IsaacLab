# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils import string as string_utils
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def track_base_height_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of base height commands using exponential kernel.
    
    This reward encourages the robot to track the target base height, enabling
    squatting behavior and expanding the operational workspace.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get current base height (z position in world frame)
    current_height = asset.data.root_pos_w[:, 2]
    # get target height command
    target_height = env.command_manager.get_command(command_name)
    # compute the error
    height_error = torch.square(target_height - current_height)
    return torch.exp(-height_error / std**2)


def track_base_height_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty for height tracking error using L2 norm.
    
    This provides a more direct penalty for height errors, encouraging faster response.
    """
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    target_height = env.command_manager.get_command(command_name)
    height_error = torch.square(target_height - current_height)
    return height_error  # Negative because it's a penalty


def track_base_height_velocity(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for moving towards target height quickly.
    
    This reward encourages the robot to change height in the correct direction
    by rewarding vertical velocity that reduces height error.
    """
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    target_height = env.command_manager.get_command(command_name)
    height_error = target_height - current_height  # Positive when need to go up
    
    # Vertical velocity (positive = going up)
    vertical_vel = asset.data.root_lin_vel_w[:, 2]
    
    # Reward moving in the correct direction
    # If error > 0 (need to go up), reward positive velocity
    # If error < 0 (need to go down), reward negative velocity
    direction_reward = height_error * vertical_vel
    
    # Scale by error magnitude to encourage faster response when error is large
    return direction_reward * torch.abs(height_error)


def track_height_knee_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    knee_joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    r"""Reward for tracking height by encouraging knee flexion/extension.
    
    This reward implements the rknee term from the paper:
    
    rknee = -||(hr,t - ht) × (qknee,t - qknee,min)/(qknee,max - qknee,min - 1/2)||
    
    where:
    - hr,t is the robot's actual height
    - ht is the target height
    - qknee,t is the current positions of robot's knee joints
    - qknee,min and qknee,max are the minimum and maximum positions of knee joints
    
    This reward encourages flexion of the knee joints when hr,t < ht,
    and encourages extension when hr,t > ht.
    
    Args:
        env: The environment.
        command_name: Name of the height command.
        knee_joint_names: List of knee joint names (regex patterns supported).
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        The reward tensor. Shape is (num_envs,).
    """
    # extract the used quantities
    asset = env.scene[asset_cfg.name]
    
    # get current base height (z position in world frame)
    current_height = asset.data.root_pos_w[:, 2]
    # get target height command
    target_height = env.command_manager.get_command(command_name)
    
    # find knee joint indices using string matching utilities
    try:
        knee_joint_ids, _ = string_utils.resolve_matching_names(knee_joint_names, asset.joint_names)
    except ValueError:
        raise ValueError(f"No knee joints found for the following joint names: {knee_joint_names}")
        # Return zero reward if no knee joints found
        return torch.zeros(env.num_envs, device=env.device)
    
    if len(knee_joint_ids) == 0:
        raise ValueError(f"No knee joints found for the following joint names: {knee_joint_names}")
        # Return zero reward if no knee joints found
        return torch.zeros(env.num_envs, device=env.device)
    
    # get knee joint positions
    knee_pos = asset.data.joint_pos[:, knee_joint_ids]  # Shape: (num_envs, num_knee_joints)
    
    # get knee joint limits
    knee_pos_limits = asset.data.default_joint_pos_limits[:, knee_joint_ids]  # Shape: (num_envs, num_knee_joints, 2)
    knee_min = knee_pos_limits[:, :, 0]  # Shape: (num_envs, num_knee_joints)
    knee_max = knee_pos_limits[:, :, 1]  # Shape: (num_envs, num_knee_joints)
    
    # compute normalized knee position: (qknee,t - qknee,min) / (qknee,max - qknee,min) - 1/2
    knee_range = knee_max - knee_min
    # avoid division by zero
    knee_range = torch.clamp(knee_range, min=1e-6)
    normalized_knee = (knee_pos - knee_min) / knee_range - 0.5  # Shape: (num_envs, num_knee_joints)
    
    # compute height error: (hr,t - ht)
    height_error = current_height - target_height  # Shape: (num_envs,)
    
    # compute reward: -||(hr,t - ht) × normalized_knee||
    # Take mean across knee joints for each environment
    reward_term = height_error.unsqueeze(-1) * normalized_knee  # Shape: (num_envs, num_knee_joints)
    reward = -torch.mean(torch.abs(reward_term), dim=1)  # Shape: (num_envs,)
    
    return reward
