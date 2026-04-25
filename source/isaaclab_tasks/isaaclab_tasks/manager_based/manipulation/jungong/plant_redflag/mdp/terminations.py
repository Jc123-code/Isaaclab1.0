# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def plant_redflag_success(
    env: ManagerBasedRLEnv,
    target_center: tuple[float, float, float] = (0.50823, -0.22484, 1.03),
    xy_tolerance: float = 0.10,
    z_tolerance: float = 0.02,
    wait_time: float = 0.5,
) -> torch.Tensor:
    """当 redflag 进入目标区域并连续保持 wait_time 秒后触发终止。"""

    drawer_tf_data: FrameTransformerData = env.scene["table_frame"].data
    redflag_pos_w = drawer_tf_data.target_pos_w[..., 0, :]

    center = torch.tensor(
        target_center,
        device=redflag_pos_w.device,
        dtype=redflag_pos_w.dtype,
    )

    # 各轴距离
    dx = torch.abs(redflag_pos_w[..., 0] - center[0])
    dy = torch.abs(redflag_pos_w[..., 1] - center[1])
    dz = torch.abs(redflag_pos_w[..., 2] - center[2])

    # 是否进入目标区域
    inside_region = (
        (dx <= xy_tolerance)
        & (dy <= xy_tolerance)
        & (dz <= z_tolerance)
    )

    # ---------------------------
    # 初始化 timer 状态缓存
    # ---------------------------
    timer_name = "_plant_redflag_success_timer"
    success_timer = getattr(env, timer_name, None)

    if success_timer is None or success_timer.shape != inside_region.shape:
        success_timer = torch.zeros_like(
            inside_region,
            dtype=torch.float32,
        )

    # ---------------------------
    # 处理 episode reset 边界
    # ---------------------------
    if hasattr(env, "episode_length_buf"):
        prev_len_name = "_plant_redflag_prev_episode_length_buf"
        prev_len = getattr(env, prev_len_name, None)
        cur_len = env.episode_length_buf

        if prev_len is None or prev_len.shape != cur_len.shape:
            success_timer.zero_()
        else:
            reset_ids = cur_len <= prev_len
            success_timer[reset_ids] = 0.0

        setattr(env, prev_len_name, cur_len.clone())

    # ---------------------------
    # 获取每一步 dt
    # ---------------------------
    dt = 1.0 / 60.0
    if hasattr(env, "step_dt"):
        dt = env.step_dt
    elif hasattr(env, "physics_dt"):
        dt = env.physics_dt

    # ---------------------------
    # 在区域内持续累计时间
    # ---------------------------
    success_timer[inside_region] += dt

    # 离开区域立即清零
    success_timer[~inside_region] = 0.0

    setattr(env, timer_name, success_timer)

    # 连续停留 wait_time 秒才成功
    success = success_timer >= wait_time

    return success