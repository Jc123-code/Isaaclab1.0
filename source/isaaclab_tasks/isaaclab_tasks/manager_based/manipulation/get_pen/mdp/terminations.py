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


def get_pen_success(env: ManagerBasedRLEnv, z_threshold: float = 0.1):
    # 这里的成功判定：Pen 相对重置时沿 z 轴上升 10cm（0.1m）即判定成功
    pen_z = env.scene["pen"].data.root_pos_w[..., 2]

    # 追踪每个子环境的初始 z 参考值（在每个回合第1步设置）
    if not hasattr(env, "_pen_initial_z"):
        env._pen_initial_z = pen_z.clone()

    # 由于多环境(batch)中可能仅部分需要重置, 每次步进时基于 episode_length_buf 更新初始化值
    first_step = env.episode_length_buf == 1
    if first_step.any():
        env._pen_initial_z[first_step] = pen_z[first_step]

    delta_z = pen_z - env._pen_initial_z
    success = delta_z >= z_threshold

    # debug:输出可选，可以在本地调试时打开
    # print(f"pen_z={pen_z}, init_z={env._pen_initial_z}, delta_z={delta_z}, success={success}")

    return success