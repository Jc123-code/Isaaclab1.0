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



def cup_success(
    env: ManagerBasedRLEnv,
    center_xy: tuple[float, float] = (0.5, 0.0),
    xy_radius: float = 0.08025,
    z_max_world: float = 1.075,
    min_count: int | None = None,
    success_ratio: float = 1,
    ball_names: list[str] | None = None,
):
    """
    判断任务是否成功：
    当足够数量的小球落入指定容器区域时，返回成功。

    参数说明：
    - env: 强化学习环境，包含场景中各个物体的信息
    - center_xy: 容器中心在世界坐标系下的 (x, y)
    - xy_radius: 容器在 xy 平面上的判定半径
    - z_max_world: 容器判定区域的 z 方向最大高度
    - min_count: 至少需要多少个球在容器内才算成功；如果为 None，则按 success_ratio 自动计算
    - success_ratio: 成功比例，例如 1 表示全部球都要进，0.5 表示一半即可
    - ball_names: 需要检测的小球名称列表；如果不传，则使用默认的 10 个球名

    返回：
    - 一个 shape 为 (env.num_envs,) 的布尔张量
    - 每个并行环境对应一个 True/False，表示该环境是否成功
    """

    # 如果没有传入球名列表，则使用默认的 10 个球名
    if ball_names is None:
        ball_names = [
            "ReferencePoint",
            "ReferencePoint_1",
            "ReferencePoint_2",
            "ReferencePoint_3",
            "ReferencePoint_4",
            "ReferencePoint_5",
            "ReferencePoint_6",
            "ReferencePoint_7",
            "ReferencePoint_8",
            "ReferencePoint_9",
        ]

    # 只保留当前场景中实际存在的球
    available = [name for name in ball_names if name in env.scene.keys()]

    # 如果一个球都不存在，则所有环境都判定为失败
    if len(available) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 为每个并行环境生成同样的容器中心坐标
    cx = torch.full((env.num_envs,), center_xy[0], device=env.device)
    cy = torch.full((env.num_envs,), center_xy[1], device=env.device)

    inside_list = []

    # 预先计算半径平方，避免循环中重复计算
    r2 = xy_radius * xy_radius

    # 遍历每一个存在的小球，判断它是否在容器区域内
    for name in available:
        # 获取该球在所有并行环境中的世界坐标，shape 为 (E, 3)
        # E 表示环境数量，3 表示 (x, y, z)
        ball_pos = env.scene[name].data.root_pos_w

        # 计算球与容器中心在 x、y 方向上的偏移
        dx = ball_pos[:, 0] - cx
        dy = ball_pos[:, 1] - cy

        # 取出球的 z 坐标
        dz = ball_pos[:, 2]

        # 判断球是否落在容器 xy 平面的圆形范围内
        inside_xy = (dx * dx + dy * dy) <= r2

        # 判断球是否低于给定的 z 高度上限
        inside_z = dz <= z_max_world

        # 同时满足 xy 和 z 条件，才算球在容器内
        inside_list.append(inside_xy & inside_z)

    # 将所有球的判定结果堆叠起来
    # inside 的 shape 为 (B, E)
    # B = 球数量，E = 环境数量
    inside = torch.stack(inside_list, dim=0)

    # 统计每个环境中有多少个球在容器内
    count_inside = inside.sum(dim=0)

    # 如果没有手动指定最少球数，则按 success_ratio 自动计算
    if min_count is None:
        # success_ratio * 球数，然后向上取整
        min_count = int(torch.ceil(torch.tensor(success_ratio * len(available))).item())

        # 至少要求 1 个球
        min_count = max(1, min_count)

    # 若每个环境中“在容器内的球数”大于等于阈值，则判定成功
    return count_inside >= min_count

