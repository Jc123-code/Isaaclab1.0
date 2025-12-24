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



def getplace_book(
    env: ManagerBasedRLEnv,
    dist_threshold: float = 0.5,
):
    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    
    # 计算 y 轴距离（使用 torch.norm 和 dim=0）
    dist_book_to_drawer = torch.norm(
        drawer_tf_data.target_pos_w[..., 0, [1]] - drawer_tf_data.target_pos_w[..., 2, [1]], 
        dim=0
    )
    dist_door_to_drawer = torch.norm(
        drawer_tf_data.target_pos_w[..., 1, [1]] - drawer_tf_data.target_pos_w[..., 2, [1]], 
        dim=0
    )
    
    # 总距离
    total_dist = dist_book_to_drawer + dist_door_to_drawer
    
    # 判断成功（可以用 logical_and 或直接比较）
    success = total_dist < dist_threshold
    # print(f"Book distance: {dist_book_to_drawer}, Door distance: {dist_door_to_drawer}, Total: {total_dist}")
    return success




