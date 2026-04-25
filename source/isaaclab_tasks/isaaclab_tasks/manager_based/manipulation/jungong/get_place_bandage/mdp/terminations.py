# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def getplace_bandage_success(env: ManagerBasedRLEnv, dist_threshold: float = 0.8) -> torch.Tensor:
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    
    # 计算各轴距离
    dist_x = torch.norm(drawer_tf_data.target_pos_w[..., 0, [0]] - drawer_tf_data.target_pos_w[..., 1, [0]], dim=-1)
    dist_y = torch.norm(drawer_tf_data.target_pos_w[..., 0, [1]] - drawer_tf_data.target_pos_w[..., 1, [1]], dim=-1)
    dist_z = torch.norm(drawer_tf_data.target_pos_w[..., 0, [2]] - drawer_tf_data.target_pos_w[..., 1, [2]], dim=-1)
    
    # 三个条件
    x_condition = dist_x < 0.3
    y_condition = dist_y < dist_threshold
    z_condition = dist_z < 0.08
    
    # 组合条件
    success = x_condition & y_condition & z_condition
    
    # print(f"X Distance: {dist_x}, Y Distance: {dist_y}, Z Distance: {dist_z}, Success: {success}")
    
    return success