# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    cleantable_ik_abs_env_cfg,
    cleantable_joint_pos_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Clean-Table-Franka-Bimanual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": cleantable_joint_pos_env_cfg.FrankaCleanTableEnvCfg,
    },
    disable_env_checker=True,
)

 

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Clean-Table-Franka-Bimanual-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": cleantable_ik_abs_env_cfg.FrankaCleanTableEnvCfg, # 环境入口
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),  # 算法入口
        "robomimic_diffusion_policy_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/dp_unet_image.json"),  # 算法入口
        "robomimic_act_cfg_entry_point": "/home/abc/IsaacLab/robomimic/robomimic/exps/templates/act.json",   # act算法入口
    },
    disable_env_checker=True,
)





