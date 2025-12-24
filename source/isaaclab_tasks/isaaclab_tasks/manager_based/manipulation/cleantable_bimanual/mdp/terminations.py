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


#第一版
# def table_clean(
#     env: ManagerBasedRLEnv,
#     dist_threshold: float = 0.16,
# ):
#     drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
#     # 查看偏差
#     dist_x = torch.norm(drawer_tf_data.target_pos_w[..., 0, [0]] - drawer_tf_data.target_pos_w[..., 1, [0]],dim=0)
#     dist_y = torch.norm(drawer_tf_data.target_pos_w[..., 0, [1]] - drawer_tf_data.target_pos_w[..., 1, [1]],dim=0)
#     dist_z = torch.norm(drawer_tf_data.target_pos_w[..., 0, [2]] - drawer_tf_data.target_pos_w[..., 1, [2]],dim=0)
#     # print(dist_x,dist_y,dist_z)
#     opened = torch.logical_and(dist_y < dist_threshold, dist_y < dist_threshold)
#     print(dist_y)
#     return opened

#第二版
# def table_clean(
#     env: ManagerBasedRLEnv,
#     dist_threshold: float = 0.1,
# ):
#     drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    
#     # 计算 paperball (index 0) 和 dustpan (index 1) 之间的3D欧几里得距离
#     paperball_pos = drawer_tf_data.target_pos_w[..., 0, :]
#     dustpan_pos = drawer_tf_data.target_pos_w[..., 1, :] 
    
#     # 计算欧几里得距离
#     dist = torch.norm(paperball_pos - dustpan_pos, dim=-1) 
    
#     # 判断是否成功（距离小于阈值）
#     success = dist < dist_threshold
    
#     print(f"Distance: {dist}")
#     return success

#第三版
# def table_clean(
#     env: ManagerBasedRLEnv,
#     dist_threshold: float = 0.1,
# ):
#     drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    
#     # 计算 paperball (index 0) 和 dustpan (index 1) 之间的3D欧几里得距离
#     # paperball_pos = drawer_tf_data.target_pos_w[..., 0, :]
#     # dustpan_pos = drawer_tf_data.target_pos_w[..., 1, :] 
    
#     dist_x = torch.norm(drawer_tf_data.target_pos_w[..., 0, [0]] - drawer_tf_data.target_pos_w[..., 1, [0]],dim=-1)
#     dist_y = torch.norm(drawer_tf_data.target_pos_w[..., 0, [1]] - drawer_tf_data.target_pos_w[..., 1, [1]],dim=-1)
#     dist_z = torch.norm(drawer_tf_data.target_pos_w[..., 0, [2]] - drawer_tf_data.target_pos_w[..., 1, [2]],dim=-1)
#     # 计算欧几里得距离
#     # dist = torch.norm(paperball_pos - dustpan_pos, dim=-1) 
    
#     # 判断是否成功（距离小于阈值）
#     success = dist_y < dist_threshold
    
#     print(f"Distance: {dist_y}")
#     return success


#第四版
# def table_clean(
#     env: ManagerBasedRLEnv,
#     dist_threshold: float = 0.1,
# ):
#     drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    
#     # 获取 paperball (index 0) 和 dustpan (index 1) 的坐标
#     paperball_x = drawer_tf_data.target_pos_w[..., 0, 0]
#     dustpan_x = drawer_tf_data.target_pos_w[..., 1, 0]

#     paperball_z = drawer_tf_data.target_pos_w[..., 0, 2]
#     dustpan_z = drawer_tf_data.target_pos_w[..., 1, 2]
    
#     # 计算各轴距离
#     dist_x = torch.norm(drawer_tf_data.target_pos_w[..., 0, [0]] - drawer_tf_data.target_pos_w[..., 1, [0]], dim=-1)
#     dist_y = torch.norm(drawer_tf_data.target_pos_w[..., 0, [1]] - drawer_tf_data.target_pos_w[..., 1, [1]], dim=-1)
#     dist_z = torch.norm(drawer_tf_data.target_pos_w[..., 0, [2]] - drawer_tf_data.target_pos_w[..., 1, [2]], dim=-1)
    
#     # 条件1: y距离小于阈值
#     y_condition = dist_y < dist_threshold
    
#     # 条件2: paperball的x坐标在dustpan的x坐标正负0.1范围内
#     x_condition = torch.abs(paperball_x - dustpan_x) < 0.1

#     z_condition = torch.abs(paperball_z - dustpan_z) < 0.1
#     # 组合条件
#     success = torch.logical_and(y_condition, x_condition, z_condition)
    
#     print(f"Y Distance: {dist_y}, X Distance: {torch.abs(paperball_x - dustpan_x)}, Success: {success}")
    
#     return success

#第五版
def table_clean(env: ManagerBasedRLEnv, dist_threshold: float = 0.1):
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    
    # 计算各轴距离
    dist_x = torch.norm(drawer_tf_data.target_pos_w[..., 0, [0]] - drawer_tf_data.target_pos_w[..., 1, [0]], dim=-1)
    dist_y = torch.norm(drawer_tf_data.target_pos_w[..., 0, [1]] - drawer_tf_data.target_pos_w[..., 1, [1]], dim=-1)
    dist_z = torch.norm(drawer_tf_data.target_pos_w[..., 0, [2]] - drawer_tf_data.target_pos_w[..., 1, [2]], dim=-1)
    
    # 三个条件
    x_condition = dist_x < 0.1
    y_condition = dist_y < dist_threshold
    z_condition = dist_z < 0.1
    
    # 组合条件
    success = x_condition & y_condition & z_condition
    
    # print(f"X Distance: {dist_x}, Y Distance: {dist_y}, Z Distance: {dist_z}, Success: {success}")
    
    return success