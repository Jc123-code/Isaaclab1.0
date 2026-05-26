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
    box_local_min: tuple[float, float, float] = (-0.12, -0.08, -0.12),
    box_local_max: tuple[float, float, float] = (0.12, 0.08, 0.14),
    wait_time: float = 0.2,
) -> torch.Tensor:
    """当全部 7.62 子弹进入 7.62 箱且全部 9mm 子弹进入 9mm 箱并持续停留后触发终止。"""

    bullet_762_names = ("bullet_762",)
    bullet_9_names = ("bullet_9",)

    ammunition_box_762: RigidObject = env.scene["Ammunition_Box_762"]
    ammunition_box_9: RigidObject = env.scene["Ammunition_Box"]

    lower = torch.tensor(
        box_local_min,
        device=env.device,
        dtype=torch.float32,
    )
    upper = torch.tensor(
        box_local_max,
        device=env.device,
        dtype=torch.float32,
    )

    debug_local_pos: dict[str, torch.Tensor] = {}

    def _all_bullets_inside_box(bullet_names: tuple[str, ...], box: RigidObject) -> torch.Tensor:
        box_pos_w = box.data.root_state_w[:, :3]
        box_quat_w = box.data.root_state_w[:, 3:7]
        inside_all = torch.ones(env.scene.num_envs, device=env.device, dtype=torch.bool)

        for bullet_name in bullet_names:
            bullet: RigidObject = env.scene[bullet_name]
            bullet_pos_w = bullet.data.root_state_w[:, :3]
            bullet_pos_box = math_utils.quat_rotate_inverse(box_quat_w, bullet_pos_w - box_pos_w)
            debug_local_pos[bullet_name] = bullet_pos_box

            inside_region = (
                (bullet_pos_box[..., 0] >= lower[0])
                & (bullet_pos_box[..., 0] <= upper[0])
                & (bullet_pos_box[..., 1] >= lower[1])
                & (bullet_pos_box[..., 1] <= upper[1])
                & (bullet_pos_box[..., 2] >= lower[2])
                & (bullet_pos_box[..., 2] <= upper[2])
            )
            inside_all &= inside_region

        return inside_all

    inside_762 = _all_bullets_inside_box(bullet_762_names, ammunition_box_762)
    inside_9 = _all_bullets_inside_box(bullet_9_names, ammunition_box_9)
    inside_region = inside_762 & inside_9

    if hasattr(env, "episode_length_buf") and env.episode_length_buf.numel() > 0:
        debug_env_id = 0
        if int(env.episode_length_buf[debug_env_id].item()) % 10 == 0:
            print(
                "[sorting_bullets_debug]"
                f" env={debug_env_id}"
                f" 9mm box={tuple(round(v, 4) for v in ammunition_box_9.data.root_state_w[debug_env_id, :3].tolist())}"
                f" 762 box={tuple(round(v, 4) for v in ammunition_box_762.data.root_state_w[debug_env_id, :3].tolist())}"
            )
            for bullet_name in bullet_9_names:
                pos = debug_local_pos[bullet_name][debug_env_id]
                print(
                    f"  {bullet_name} -> 9box local_xyz="
                    f"({pos[0].item():.4f}, {pos[1].item():.4f}, {pos[2].item():.4f})"
                )
            for bullet_name in bullet_762_names:
                pos = debug_local_pos[bullet_name][debug_env_id]
                print(
                    f"  {bullet_name} -> 762box local_xyz="
                    f"({pos[0].item():.4f}, {pos[1].item():.4f}, {pos[2].item():.4f})"
                )
            print(
                f"  inside_9={bool(inside_9[debug_env_id].item())}"
                f" inside_762={bool(inside_762[debug_env_id].item())}"
                f" all_success={bool(inside_region[debug_env_id].item())}"
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
