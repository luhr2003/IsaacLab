# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command generators for locomotion/velocity tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from .commands_cfg import UniformVelocityBaseHeightCommandCfg


class UniformVelocityBaseHeightCommand(UniformVelocityCommand):
    """Uniform SE(2) velocity command augmented with a base-height command.

    Output command is a 4D vector: [vx, vy, yaw_rate, target_height].

    Intended behavior (hybrid):
    - Walking at a fixed target height for the resampling interval.
    - Sometimes command zero velocity (stop) and sample a lower target height (squat).
    - Resume walking and sample a higher target height (stand / normal walk height).
    """

    cfg: "UniformVelocityBaseHeightCommandCfg"

    def __init__(self, cfg: "UniformVelocityBaseHeightCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.target_height = torch.zeros(env.num_envs, device=self.device)
        # --- 3-case state machine bookkeeping (walk / stand-normal / stand-squat)
        # default: treat envs as standing at normal height at the beginning
        self.prev_stand_normal = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
        self.prev_stand_squat = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.prev_walk = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Command tensor. Shape: (num_envs, 4)."""
        return torch.cat([self.vel_command_b, self.target_height.unsqueeze(-1)], dim=-1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample command with explicit 3-case transition logic.

        Cases (mutually exclusive):
        - walk:          is_standing_env == False
        - stand-normal:  is_standing_env == True and target_height >= squatting_threshold
        - stand-squat:   is_standing_env == True and target_height <  squatting_threshold

        Transition rules (same spirit as reference `commands.py`):
        - If we were walking, we don't allow a direct jump into *deep squat* in one resample
          when `random_height_during_walking` is False (walk -> stand-squat is clamped to stand-normal).
        - If we were in deep squat, we don't allow a direct transition into walking in one resample
          when `random_height_during_walking` is False (stand-squat -> walk becomes stand + resample height).
        """
        # sample vx/vy/yaw (+ standing flags) using upstream logic
        super()._resample_command(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device)

        # 1) start from default height everywhere
        self.target_height[env_ids_t] = self.cfg.default_height

        # 2) sample target height for standing envs with explicit 2-way split:
        #    - stand-squat  : [base_height_low, squatting_threshold)
        #    - stand-normal : [squatting_threshold, base_height_high]
        stand_mask = self.is_standing_env[env_ids_t]
        if stand_mask.any():
            stand_ids = env_ids_t[stand_mask]
            n = stand_ids.numel()
            low = float(self.cfg.ranges.base_height[0])
            high = float(self.cfg.ranges.base_height[1])
            squat_hi = float(self.cfg.squatting_threshold)
            squat_hi = max(min(squat_hi, high), low)

            # choose squat vs normal for standing envs
            choose_squat = torch.rand((n,), device=self.device) < float(self.cfg.lower_height_bias)

            if choose_squat.any():
                ids = stand_ids[choose_squat]
                self.target_height[ids] = self._uniform(ids.numel(), low, squat_hi)
            if (~choose_squat).any():
                ids = stand_ids[~choose_squat]
                self.target_height[ids] = self._uniform(ids.numel(), squat_hi, high)

        # 3) optionally sample target height for walking envs (walk at specific height range)
        if self.cfg.random_height_during_walking:
            walk_mask = ~stand_mask
            if walk_mask.any():
                walk_ids = env_ids_t[walk_mask]
                low = float(self.cfg.walk_height_min)
                high = float(self.cfg.ranges.base_height[1])
                low = min(max(low, float(self.cfg.ranges.base_height[0])), high)
                self.target_height[walk_ids] = self._uniform(walk_ids.numel(), low, high)

        # 4) compute current 3-case state for these envs
        squat_thr = self.cfg.squatting_threshold
        current_stand_normal = self.is_standing_env & (self.target_height >= squat_thr)
        current_stand_squat = self.is_standing_env & (self.target_height < squat_thr)
        current_walk = ~self.is_standing_env

        # 5) apply transition constraints (only when walking height is NOT randomized)
        if not self.cfg.random_height_during_walking:
            # Case A: prev walk -> current stand-squat : clamp to stand-normal height
            walk_to_squat = self.prev_walk[env_ids_t] & current_stand_squat[env_ids_t]
            if walk_to_squat.any():
                ids = env_ids_t[walk_to_squat]
                self.target_height[ids] = self.cfg.default_height
                current_stand_squat[ids] = False
                current_stand_normal[ids] = True

            # Case B: prev stand-squat -> current walk : force standing + resample height
            squat_to_walk = self.prev_stand_squat[env_ids_t] & current_walk[env_ids_t]
            if squat_to_walk.any():
                ids = env_ids_t[squat_to_walk]
                self.is_standing_env[ids] = True
                # re-sample standing height with explicit split (same as step 2)
                n = ids.numel()
                low = float(self.cfg.ranges.base_height[0])
                high = float(self.cfg.ranges.base_height[1])
                squat_hi = float(self.cfg.squatting_threshold)
                squat_hi = max(min(squat_hi, high), low)
                choose_squat = torch.rand((n,), device=self.device) < float(self.cfg.lower_height_bias)
                if choose_squat.any():
                    self.target_height[ids[choose_squat]] = self._uniform(int(choose_squat.sum()), low, squat_hi)
                if (~choose_squat).any():
                    self.target_height[ids[~choose_squat]] = self._uniform(int((~choose_squat).sum()), squat_hi, high)
                # recompute local state for those ids
                current_walk[ids] = False
                current_stand_squat[ids] = self.target_height[ids] < squat_thr
                current_stand_normal[ids] = ~current_stand_squat[ids]

        # 6) if crouch-walking too low, scale velocity down for stability
        if self.cfg.random_height_during_walking:
            walk_ids_all = env_ids_t[(~stand_mask)]
            if walk_ids_all.numel() > 0:
                crouch_mask = self.target_height[walk_ids_all] < self.cfg.min_walk_height
                if crouch_mask.any():
                    crouch_ids = walk_ids_all[crouch_mask]
                    low = self.cfg.ranges.base_height[0]
                    high = self.cfg.min_walk_height
                    scale = 1.0 - (high - self.target_height[crouch_ids]) / max(high - low, 1e-6)
                    scale = torch.clamp(scale, 0.0, 1.0)
                    self.vel_command_b[crouch_ids, :] *= scale.unsqueeze(1)

        # 7) update previous-state trackers for next resample
        self.prev_stand_normal[env_ids_t] = current_stand_normal[env_ids_t]
        self.prev_stand_squat[env_ids_t] = current_stand_squat[env_ids_t]
        self.prev_walk[env_ids_t] = current_walk[env_ids_t]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset command generator and its transition state."""
        extras = super().reset(env_ids)
        if env_ids is None:
            self.prev_stand_normal[:] = True
            self.prev_stand_squat[:] = False
            self.prev_walk[:] = False
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device)
            self.prev_stand_normal[env_ids_t] = True
            self.prev_stand_squat[env_ids_t] = False
            self.prev_walk[env_ids_t] = False
        return extras

    def _update_command(self) -> None:
        """Post-process command.

        Key behavior difference from IsaacLab's `UniformVelocityCommand`:
        - For "standing" envs we enforce **vx=vy=0** but we DO NOT force yaw to zero,
          so the robot can **turn in place**.
        - If heading control is enabled, yaw-rate is computed from heading error (same as upstream).
        """
        # Compute angular velocity from heading direction (same as IsaacLab)
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if env_ids.numel() > 0:
                heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
                self.vel_command_b[env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )

        # Enforce "standing" => no translation, but allow yaw for in-place turning
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if standing_env_ids.numel() > 0:
            self.vel_command_b[standing_env_ids, 0:2] = 0.0

    def _sample_height(self, n: int) -> torch.Tensor:
        """Sample target heights.

        If biasing is enabled, sample low heights with probability `lower_height_bias`
        from [min, sample_middle_height], else from [sample_middle_height, max].
        """
        lo, hi = self.cfg.ranges.base_height
        if n <= 0:
            return torch.empty((0,), device=self.device)

        if not self.cfg.bias_height_randomization:
            return torch.empty((n,), device=self.device).uniform_(lo, hi)

        # biased sampling
        mid = self.cfg.sample_middle_height if self.cfg.sample_middle_height is not None else 0.5 * (lo + hi)
        mid = float(torch.clamp(torch.tensor(mid), min=lo, max=hi).item())
        use_lower = torch.rand((n,), device=self.device) < self.cfg.lower_height_bias
        out = torch.empty((n,), device=self.device)
        if use_lower.any():
            out[use_lower] = torch.empty((int(use_lower.sum()),), device=self.device).uniform_(lo, mid)
        if (~use_lower).any():
            out[~use_lower] = torch.empty((int((~use_lower).sum()),), device=self.device).uniform_(mid, hi)
        return out

    def _uniform(self, n: int, lo: float, hi: float) -> torch.Tensor:
        """Sample uniformly in [lo, hi]. If hi <= lo, returns a constant lo tensor."""
        if n <= 0:
            return torch.empty((0,), device=self.device)
        if hi <= lo:
            return torch.full((n,), lo, device=self.device)
        return torch.empty((n,), device=self.device).uniform_(lo, hi)


