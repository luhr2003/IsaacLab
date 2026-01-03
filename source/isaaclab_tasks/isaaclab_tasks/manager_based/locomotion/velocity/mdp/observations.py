# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helper terms for hybrid velocity+height commands."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def base_height_from_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Extract base-height (last dim) from a hybrid command vector.

    Assumes command shape is (num_envs, 4): [vx, vy, yaw_rate, height].
    Returns shape (num_envs, 1) for concatenation with other observations.
    """
    cmd = env.command_manager.get_command(command_name)
    if cmd.dim() == 1:
        # (num_envs,) -> (num_envs, 1)
        return cmd.unsqueeze(-1)
    return cmd[:, 3:4]


def base_velocity_from_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Extract SE(2) velocity part from a hybrid command vector.

    Assumes command shape is (num_envs, 4): [vx, vy, yaw_rate, height].
    Returns shape (num_envs, 3).
    """
    cmd = env.command_manager.get_command(command_name)
    if cmd.dim() == 1:
        # fall back: treat scalar command as zero velocity (shouldn't happen for hybrid commands)
        return torch.zeros((env.num_envs, 3), device=env.device)
    return cmd[:, :3]


def go2_unified_height_command(
    env: ManagerBasedRLEnv,
    *,
    velocity_command_name: str = "base_velocity",
    height_command_name: str = "base_height",
) -> torch.Tensor:
    """Return a (num_envs, 1) height command for both task styles:

    - **FlatHeight** style: height is a separate scalar command term named `base_height`.
    - **Hybrid 4D** style: height is embedded as the 4th component of `base_velocity`.

    This makes observation shapes identical across tasks.
    """
    # Prefer a dedicated height command if present (flat-height envs).
    try:
        h = env.command_manager.get_command(height_command_name)
        if h.dim() == 1:
            return h.unsqueeze(-1)
        # already (num_envs, 1)
        return h
    except Exception:
        pass

    # Fallback: embedded in hybrid command vector (vx, vy, yaw, height).
    cmd = env.command_manager.get_command(velocity_command_name)
    if cmd.dim() == 1:
        return cmd.unsqueeze(-1)
    return cmd[:, 3:4]


