# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for custom command generators used by locomotion/velocity tasks."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .commands import UniformVelocityBaseHeightCommand


@configclass
class UniformVelocityBaseHeightCommandCfg(UniformVelocityCommandCfg):
    """Uniform SE(2) velocity command with an additional base-height command."""

    class_type: type = UniformVelocityBaseHeightCommand

    # default walking height for envs that are not sampling height
    default_height: float = 0.30

    # if True, also sample height while walking (enables "walk at a specific crouched height")
    random_height_during_walking: bool = True

    # below this height, reduce commanded walking speed for stability
    min_walk_height: float = 0.2

    # minimum base height when sampling height during walking (i.e. "walk at a specific height range")
    # walking height will be sampled from [walk_height_min, ranges.base_height[1]]
    walk_height_min: float = 0.15

    # threshold to classify "standing at squat height" vs "standing at normal height"
    # (used only for transition logic / state machine)
    squatting_threshold: float = 0.2

    # height sampling (biased towards squat heights if enabled)
    bias_height_randomization: bool = True
    lower_height_bias: float = 0.7
    sample_middle_height: float | None = None

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        base_height: tuple[float, float] = MISSING
        """Range for base height command (in m)."""

    ranges: Ranges = MISSING


