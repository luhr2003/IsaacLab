# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_height_env_cfg import UnitreeGo2RoughHeightEnvCfg
from .unified_obs_cfg import Go2UnifiedObservationsCfg


@configclass
class UnitreeGo2FlatHeightEnvCfg(UnitreeGo2RoughHeightEnvCfg):
    # make observations identical to other Go2 velocity/height tasks (order + dimension)
    observations: Go2UnifiedObservationsCfg = Go2UnifiedObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # no heading control: use direct yaw-rate (rz) control via ang_vel_z
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_control_stiffness = 0.0
        # remove heading range to avoid warnings when heading_command=False
        self.commands.base_velocity.ranges.heading = None

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2FlatHeightEnvCfg_PLAY(UnitreeGo2FlatHeightEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

