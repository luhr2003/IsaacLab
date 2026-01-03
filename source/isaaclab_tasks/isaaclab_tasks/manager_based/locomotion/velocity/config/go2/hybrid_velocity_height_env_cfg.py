# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 hybrid velocity+height task.

Hybrid behaviors enabled by the command generator:
- Walk at a specified target height (including crouch-walk).
- Stop and squat lower (vx/vy ~ 0, low target height).
- Stand up and continue walking.
"""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_height_env_cfg import LocomotionVelocityHeightRoughEnvCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from .unified_obs_cfg import Go2UnifiedObservationsCfg


@configclass
class Go2HybridCommandsCfg:
    """Single hybrid command: SE(2) velocity + base height."""

    base_velocity = mdp.UniformVelocityBaseHeightCommandCfg(
        asset_name="robot",
        # resample fairly often so the agent learns transitions (walk->stop->squat->stand->walk)
        resampling_time_range=(1.0, 2.5),
        # target mixture (approx): walk 50%, stand-normal 20%, stand-squat 30%
        # => standing total = 0.5; squat|standing = 0.3/0.5 = 0.6
        rel_standing_envs=0.50,
        # no heading control: use direct yaw-rate (rz) control via ang_vel_z
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        default_height=0.30,
        random_height_during_walking=True,
        min_walk_height=0.2,
        walk_height_min=0.15,
        bias_height_randomization=True,
        lower_height_bias=0.60,
        sample_middle_height=0.30,
        ranges=mdp.UniformVelocityBaseHeightCommandCfg.Ranges(
            # small but non-zero velocities
            lin_vel_x=(-1, 1.0),
            lin_vel_y=(-1, 1),
            ang_vel_z=(-1, 1),
            heading=None,
            # [stand high .. squat low]
            base_height=(0.05, 0.5),
        ),
    )


@configclass
class Go2HybridRewardsCfg:
    """Rewards: velocity tracking + height tracking from hybrid command."""

    # track vx/vy/yaw (functions read only first 3 dims, so hybrid cmd is compatible)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.20)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.20)}
    )

    # height tracking (extract height from 4th dim)
    track_base_height_exp = RewTerm(
        func=mdp.track_base_height_exp_from_command,
        weight=2.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.05)},
    )
    track_base_height_l2 = RewTerm(
        func=mdp.track_base_height_l2_from_command,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )
    track_base_height_velocity = RewTerm(
        func=mdp.track_base_height_velocity_from_command,
        weight=0.3,
        params={"command_name": "base_velocity"},
    )
    # knee shaping (Go2 knee joint is calf_joint)
    track_height_knee = RewTerm(
        func=mdp.track_height_knee_reward_from_command,
        weight=1.0,
        params={"command_name": "base_velocity", "knee_joint_names": [".*_calf_joint"]},
    )

    # penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class UnitreeGo2FlatHybridVelocityHeightEnvCfg(LocomotionVelocityHeightRoughEnvCfg):
    """Go2 hybrid velocity+height environment (flat terrain)."""

    # override components
    commands: Go2HybridCommandsCfg = Go2HybridCommandsCfg()
    observations: Go2UnifiedObservationsCfg = Go2UnifiedObservationsCfg()
    rewards: Go2HybridRewardsCfg = Go2HybridRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # robot
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # reduce action scale for stability during crouch transitions
        self.actions.joint_pos.scale = 0.25

        # flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # faster control loop (50Hz controller at 200Hz physics)
        self.sim.dt = 1 / 200
        self.decimation = 4
        self.sim.render_interval = self.decimation

        # keep task clean: no pushes/external wrench
        self.events.push_robot = None
        self.events.base_external_force_torque = None

        # reset: don't add random base velocity (reduce drift during squat)
        if self.events.reset_base is not None:
            self.events.reset_base.params["velocity_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }


@configclass
class UnitreeGo2FlatHybridVelocityHeightEnvCfg_PLAY(UnitreeGo2FlatHybridVelocityHeightEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None


