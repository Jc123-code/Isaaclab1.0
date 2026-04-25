# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    getplace_bandage_ik_abs_env_cfg,
    getplace_bandage_joint_pos_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Get-place-Bandage-Franka-Bimanual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": getplace_bandage_joint_pos_env_cfg.FrankaGetPlacebandageEnvCfg,
    },
    disable_env_checker=True,
)

 

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Get-place-Bandage-Franka-Bimanual-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": getplace_bandage_ik_abs_env_cfg.FrankaGetPlacebandageEnvCfg, # 环境入口
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),  # 算法入口
        "robomimic_diffusion_policy_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/dp_unet_image.json"),  # 算法入口
        "robomimic_act_cfg_entry_point": "/home/abc/IsaacLab/robomimic/robomimic/exps/templates/act.json",   # act算法入口
    },
    disable_env_checker=True,
)

# # 兼容大小写/漏版本号的调用别名
# gym.register(
#     id="Isaac-Get-place-bandage-Franka-Bimanual-IK-Abs",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": getplace_bandage_ik_abs_env_cfg.FrankaGetPlacebandageEnvCfg,
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
#         "robomimic_diffusion_policy_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/dp_unet_image.json"),
#         "robomimic_act_cfg_entry_point": "/home/abc/IsaacLab/robomimic/robomimic/exps/templates/act.json",
#     },
#     disable_env_checker=True,
# )




