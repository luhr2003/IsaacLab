# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 stand+squat task with 4D hybrid command: [vx, vy, yaw, height].

Requirements supported:
- Command is always 4D: [vx, vy, yaw_rate, target_height]
- No heading control (use direct yaw-rate / rz control)
- Turn-in-place supported (standing => vx=vy=0 but yaw is allowed)
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_height_env_cfg import LocomotionVelocityHeightRoughEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from .unified_obs_cfg import Go2UnifiedObservationsCfg


@configclass
class Go2StandSquatCommandsCfg:
    """Commands consistent with `flat_height_env_cfg.py`:
    - `base_velocity`: (vx, vy, rz) with small ranges
    - `base_height`: separate height command with larger amplitude
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.8, 1.6),
        rel_standing_envs=0.0,
        heading_command=False,  # use direct rz
        rel_heading_envs=0.0,
        heading_control_stiffness=0.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.50, 0.50),
            heading=None,
        ),
    )

    base_height = mdp.UniformHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.20, 0.60),
        # normal standing = 30%, squat sampling = 70%
        rel_squat_envs=0.70,
        default_standing_height=0.30,
        ranges=mdp.UniformHeightCommandCfg.Ranges(height=(0.05, 0.50)),
    )


@configclass
class Go2StandSquatRewardsCfg:
    """Rewards tuned for stand/squat height tracking (with separate height command)."""

    # velocity tracking (works with 4D command; functions slice [:, :2] and [:, 2])
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.8, params={"command_name": "base_velocity", "std": math.sqrt(0.20)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.6, params={"command_name": "base_velocity", "std": math.sqrt(0.20)}
    )

    # height tracking from separate height command
    track_base_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=3.0,
        params={"command_name": "base_height", "std": math.sqrt(0.03)},
    )
    track_base_height_l2 = RewTerm(
        func=mdp.track_base_height_l2,
        weight=-3.0,
        params={"command_name": "base_height"},
    )
    track_base_height_velocity = RewTerm(
        func=mdp.track_base_height_velocity,
        weight=0.35,
        params={"command_name": "base_height"},
    )

    # Encourage knee flexion/extension consistent with target height (Go2 uses calf_joint as knee).
    track_height_knee = RewTerm(
        func=mdp.track_height_knee_reward,
        weight=1.0,
        params={"command_name": "base_height", "knee_joint_names": [".*_calf_joint"]},
    )

    # Allow vertical movement (otherwise squat/stand becomes sluggish)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.15)

    # Allow faster action changes for responsiveness
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)


@configclass
class UnitreeGo2FlatStandSquatHeightEnvCfg(LocomotionVelocityHeightRoughEnvCfg):
    """Go2 stand+squat using a single 4D hybrid command."""

    commands: Go2StandSquatCommandsCfg = Go2StandSquatCommandsCfg()
    observations: Go2UnifiedObservationsCfg = Go2UnifiedObservationsCfg()
    rewards: Go2StandSquatRewardsCfg = Go2StandSquatRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # reduce action scale for stability
        self.actions.joint_pos.scale = 0.25

        # flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan for this task
        self.scene.height_scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # faster control loop for "real-time" response (50Hz @ 200Hz)
        self.decimation = 4
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation

        # keep task clean
        self.events.push_robot = None
        self.events.base_external_force_torque = None


@configclass
class UnitreeGo2FlatStandSquatHeightEnvCfg_PLAY(UnitreeGo2FlatStandSquatHeightEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None


