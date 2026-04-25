# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sorting_bullets_success(
    env: ManagerBasedRLEnv,
    box_local_min: tuple[float, float, float] = (-0.07, -0.05, -0.03),
    box_local_max: tuple[float, float, float] = (0.07, 0.05, 0.06),
    wait_time: float = 0.5,
) -> torch.Tensor:
    """当 bullet_9 进入 Ammunition_Box 并连续保持 wait_time 秒后触发终止。"""

    bullet_9: RigidObject = env.scene["bullet_9"]
    ammunition_box: RigidObject = env.scene["Ammunition_Box"]

    bullet_pos_w = bullet_9.data.root_state_w[:, :3]
    box_pos_w = ammunition_box.data.root_state_w[:, :3]
    box_quat_w = ammunition_box.data.root_state_w[:, 3:7]

    # 将子弹位置转换到弹药箱局部坐标系下，再判断是否进入箱体内部区域。
    bullet_pos_box = math_utils.quat_rotate_inverse(box_quat_w, bullet_pos_w - box_pos_w)

    lower = torch.tensor(
        box_local_min,
        device=bullet_pos_box.device,
        dtype=bullet_pos_box.dtype,
    )
    upper = torch.tensor(
        box_local_max,
        device=bullet_pos_box.device,
        dtype=bullet_pos_box.dtype,
    )

    inside_region = (
        (bullet_pos_box[..., 0] >= lower[0])
        & (bullet_pos_box[..., 0] <= upper[0])
        & (bullet_pos_box[..., 1] >= lower[1])
        & (bullet_pos_box[..., 1] <= upper[1])
        & (bullet_pos_box[..., 2] >= lower[2])
        & (bullet_pos_box[..., 2] <= upper[2])
    )

    # ---------------------------
    # 初始化 timer 状态缓存
    # ---------------------------
    timer_name = "_sorting_bullets_success_timer"
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
        prev_len_name = "_sorting_bullets_prev_episode_length_buf"
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
