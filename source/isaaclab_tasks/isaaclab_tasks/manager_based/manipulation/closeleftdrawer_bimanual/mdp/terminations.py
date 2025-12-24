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



def leftdrawer_closed(
    env: ManagerBasedRLEnv,
    dist_threshold: float = 0.18,
):
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    # 查看偏差
    dist_x = torch.norm(drawer_tf_data.target_pos_w[..., 0, [0]] - drawer_tf_data.target_pos_w[..., 1, [0]],dim=0)
    dist_y = torch.norm(drawer_tf_data.target_pos_w[..., 0, [1]] - drawer_tf_data.target_pos_w[..., 1, [1]],dim=0)
    dist_z = torch.norm(drawer_tf_data.target_pos_w[..., 0, [2]] - drawer_tf_data.target_pos_w[..., 1, [2]],dim=0)
    # print(dist_x,dist_y,dist_z)
    opened = torch.logical_and(dist_x > dist_threshold, dist_x > dist_threshold)
    print(dist_x)
    return opened

# def leftdrawer_closed(
#     env: ManagerBasedRLEnv,
#     dist_threshold: float = 0.8,
# ):
#     """判断左抽屉是否关闭（仅考虑x方向距离）"""
#     drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    
#     # 计算x方向距离
#     dist_x = torch.abs(
#         drawer_tf_data.target_pos_w[..., 0, 0] - drawer_tf_data.target_pos_w[..., 1, 0]
#     )
    
#     # 判断是否关闭
#     closed = dist_x < dist_threshold
#     print(dist_x)
#     return closed


